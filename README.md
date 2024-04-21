# LLM-baselines

Repository for my participation in the [LLM training and architecture exploration hackathon](https://lu.ma/lauzhack-llms-apis) organized by [LauzHack](https://lauzhack.com/) and the [MLO lab](https://www.epfl.ch/labs/mlo/).

The ideas is to train the best performing language model on [Slimpajama](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) with 3 hours of compute on a A100. You have to start with a fairly simple codebase: [llm-baselines](https://github.com/epfml/llm-baselines).

# Idea and approach

Two separate ideas:


## Better optimizer (failed)

- I tried the (infamous) second-order optimizer [Sophia](https://arxiv.org/abs/2305.14342)
    - It didn't make a meaningiful difference during training, even at multiple model scales and architectures. Conclusion: it was slower (both in terms of throughput and final performance) than just using fused AdamW.

## Bitter lesson, GPUs go brrrr (successful)

After launching many runs, I came to the same conclusion as [James Bekter](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/). All the training runs were very similar. Thus, in our compute-bound regime, the best thing to do is to have the most efficient model (with enough capacity) and just train on the maximum of tokens possible in 3 hours.


#### Architecure
- After some testing, I decided to go with llama architecure instead of classic gpt-2 architecture
    - why?
        - [RoPE](https://arxiv.org/abs/2104.09864) instead of learned embeddings, less parameters to learn, and works well in practice.
        - SwiGLU (Noam Shazeer showed  it was better, let's say I trust the dude)
        - RMSNrom instead of LayerNorm, because speed

        
#### Changes from default repo 
- grad_clip = 1.0 (good practice, helps with stability)
- Doubled the effective batch size (micro batch size at 64, gradient_accumulation=4), better learning dynamics and higher throughput (10% increase)
- Original Llama implementation didn't torch.compile() because of the usage of complex numbers in RoPE, thus I reimplemented RoPE without them (thanks to [lucidrains](https://github.com/lucidrains/rotary-embedding-torch/tree/main)). Now it compiles! (no drawbacks, 15% increase in throughput). 
- Reimplemented RMS norm, so that normalization is done in fp32 (important because bf16 precision is too low)


#### How to choose model size

Now that we've optimized the throughput, given we're compute-bound, we should think about [Chinchilla](https://arxiv.org/abs/2203.15556)!

- The model size should be proportional to how many tokens we can crush through in 3 hours of training. Chinchilla concluded that the rule of thumb is 20:1 (20 tokens for 1 parameter) for a > 400M param model, however at our scale, the confidence interval for this ratio is probably quite large.

- For our final model (177M parameters), we have a throughput of 100k tokens/s. With 3 hours of training, this gives us approximately 100'000 * 3 * 60 * 60 = 1.08B tokens. This gives us a ratio of about 6:1. This is not close to Chinchilla, however that's what we found to work the best. Current theory as to why there's a gap is because the effective batch I used was smaller compared to Chinchilla's.

<br>
    
- Two choices are possible to tweak the number of parameters:
    - Deeper: number of layers
    - Wider: number of heads (must be changed with embedding dimension) such that emb_dim/n_head mod 64 = 0 (so that our A100 keeps going brr, using the tf32 tensor cores)

- In the end, going wider was the most throughput efficient method to increase parameter count, and this directly translated into better results. 
- The final choice is: n_layers=10, n_heads=16, emb_dim = 1024

        


# Final model weights

**WARNING: my final "optimal" run crashed 30 mins before the submission. I'd be grateful if you could retrain from scratch and check the results then.**

However, for the sake of having weights, please find them here: https://drive.google.com/file/d/1zj0gXE1s0WU9ToTmKwOi0FCs2d3XVoxk/view?usp=drive_link.


I'll update this doc and put the true final model weights in here (when it's done training strictly for 3 hours), but that will be after the submission deadline, so feel free to ignore the link if you think that this could be cheating.



# Training script and config

### Dependencies


Install dependencies: 

```
pip install -r requirements.txt
```

(Packages added to base docker image: `ipykernel`, `einops`, `beartype`)


### Reproducing the experiment


```sh
python src/main.py --config src/config/final/noam_wide3.yaml 
```

