"""
Llama style Language Model.
References:
1) Llama inference code:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
2) Mistral one file ref:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
3) Llama paper:
https://arxiv.org/pdf/2302.13971.pdf
 
Main differences from GPT2:
* Uses RMSNorm instead of LayerNorm
* Uses a slightly different MLP (SwiGLU)
* rotary embeddings (RoPE)
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import CausalSelfAttention, GPTBase

from .common import RMSNorm

from .rotary_embedding import RotaryEmbedding

# dim = self.head_dim, end = config.sequence_length)

# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     return torch.polar(torch.ones_like(freqs), freqs)  # complex64


# def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     """
#     freqs_cis: complex - (seq_len, head_dim / 2)
#     x: complex - (bsz, seq_len, head_dim / 2)
#     """
#     ndim = x.ndim
#     assert 1 < ndim
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
#         freqs_cis.shape,
#         (x.shape[1], x.shape[-1]),
#     )
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(*shape)


# def apply_rotary_emb(q, k, freqs_cis):
#     # q, k: (B, T, nh, hs)
#     # freq_cis: (T, hs)
#     # return: (B, T, nh, hs), (B, T, nh, hs)
#     q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
#     k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
#     freqs_cis = _reshape_for_broadcast(freqs_cis, q_)
#     xq_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
#     return xq_out.type_as(q), xk_out.type_as(k)







class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class LlamaAttention(CausalSelfAttention):


    def __init__(self, config, rotary_emb):
        super().__init__(config)
        self.rotary_emb = rotary_emb


    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nh, T,  hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)


#        q, k = apply_rotary_emb(q, k, freqs_cis)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class LlamaBlock(nn.Module):
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = LlamaAttention(config, rotary_emb)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Noam(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.valid_token_ids = set(self.tokenizer._special_tokens.values()).union(range(self.tokenizer.n_vocab))

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        ## quick fix
        self.rotary_emb.freqs = self.rotary_emb.freqs.cuda()

        #self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([LlamaBlock(config, self.rotary_emb) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if self.config.weight_tying:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying



        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        #freqs_cis = self.freqs_cis.to(x.device)[pos]

        for block_idx, block in enumerate(self.transformer.h):
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = True,
        return_text: bool = True,
    ):
        """
        Args:
            prompt (str): Input text prompt.
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Sampling temperature (higher = more random).
            top_k (int or None): Top-k sampling (keep only top k tokens).
            top_p (float or None): Top-p (nucleus) sampling (keep tokens with cumulative prob >= p).
            do_sample (bool): Whether to sample or use greedy decoding.
            return_text (bool): If True, decode tokens to string.
    
        Returns:
            str or List[int]: Generated text (if return_text) or token IDs.
        """
        # Tokenize input prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.sequence_length :]  # crop to block size
            out = self.forward(idx_cond, get_logits=True)
            logits = out["logits"]  # shape (B, 1, vocab_size)
            logits = logits[:, -1, :] / temperature
    
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float("-inf")
    
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
    
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1]
                sorted_mask[:, 0] = 0
    
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = float("-inf")
    
            probs = F.softmax(logits, dim=-1)
    
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
    
            input_ids = torch.cat((input_ids, next_token), dim=1)
    
        if return_text:
            # The model outputs token_ids > vocab_size some time, e.g. 50269. TODO: investigate
            output_text = self.tokenizer.decode([tok for tok in input_ids[0].tolist() if tok in self.valid_token_ids])
            output_text = output_text.replace("<|endoftext|>", "").strip()
            return output_text
        else:
            return input_ids[0].tolist()
