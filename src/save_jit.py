import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb

import config
from models.utils import get_model
from data.utils import get_dataset
from optim.base import train_base
from optim.sofia import SophiaG
import distributed
from torch.optim.lr_scheduler import OneCycleLR

from peft import LoKrConfig, LoraConfig, LoHaConfig, get_peft_model

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def main(args):

    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = 'cpu'
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")

    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
    if args.data_in_ram:
        data = {'train': np.array(data['train']), 'val': np.array(data['val'])}

    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")

    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'

    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0

    args.world_size = distributed_backend.get_world_size()
    exp_name = args.exp_name

    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy, group=args.wandb_group)

    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
        distributed_backend.sync()
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")): # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    itr = 0
    rng_state_dict = None
    if args.use_pretrained == "auto":
        checkpoints = [file for file in os.listdir(ckpt_path) if 'ckpt_' in file]
        if checkpoints:
            args.use_pretrained = sorted(checkpoints)[-1]
        else:
            args.use_pretrained = None

    if args.use_pretrained is not None and 'ckpt_' in args.use_pretrained:
        last_ckpt_path = args.use_pretrained
        print(f"Resuming from {last_ckpt_path}")
        # checkpoint = torch.load(os.path.join(ckpt_path, last_ckpt_path))
        checkpoint = torch.load(last_ckpt_path, weights_only=True)
        print(checkpoint)

        model_state_dict = {distributed_backend.translate_model_parameter_name_for_node(k.replace("_orig_mod.", ""))[0]:v for k,v in checkpoint['model'].items()}
        # FIXME checkpoints from compiled model have _orig_mod keyword

        optimizer_state_dict = checkpoint['optimizer']
        rng_state_dict = {
            module: checkpoint[module] for module in [
                "cpu_rng_state",
                "gpu_rng_state",
                "numpy_rng_state",
                "py_rng_state",
                "train_sampler_state"
            ]
        }

        model.load_state_dict(model_state_dict)
        opt.load_state_dict(optimizer_state_dict)
        itr = checkpoint['itr']
        if scheduler is not None:
            scheduler_state_dict = checkpoint['scheduler']
            scheduler.load_state_dict(scheduler_state_dict)

    elif args.use_pretrained is not None:
        last_ckpt_path = args.use_pretrained
        print(f"Load from {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, weights_only=True)

        model_state_dict = {distributed_backend.translate_model_parameter_name_for_node(k.replace("_orig_mod.", ""))[0]:v for k,v in checkpoint['model'].items()}
        # FIXME checkpoints from compiled model have _orig_mod keyword

        # only load the model; ignore scheduler, optimizer, itr
        model.load_state_dict(model_state_dict)

    if args.model in ['base', 'llama2', 'noam']: # all train functions have the same interface
        train = train_base
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    # Apply LoRA
    """
    Noam(
      (transformer): ModuleDict(
        (wte): Embedding(50304, 1024)
        (drop): Dropout(p=0.0, inplace=False)
        (h): ModuleList(
          (0-9): 10 x LlamaBlock(
            (ln_1): RMSNorm()
            (attn): LlamaAttention(
              (c_attn): Linear(in_features=1024, out_features=3072, bias=False)
              (c_proj): Linear(in_features=1024, out_features=1024, bias=False)
              (attn_dropout): Dropout(p=0.0, inplace=False)
              (resid_dropout): Dropout(p=0.0, inplace=False)
              (rotary_emb): RotaryEmbedding()
            )
            (ln_2): RMSNorm()
            (mlp): LlamaMLP(
              (w1): Linear(in_features=1024, out_features=2816, bias=False)
              (w2): Linear(in_features=1024, out_features=2816, bias=False)
              (c_proj): Linear(in_features=2816, out_features=1024, bias=False)
            )
          )
        )
        (ln_f): RMSNorm()
      )
      (lm_head): Linear(in_features=1024, out_features=50304, bias=False)
      (rotary_emb): RotaryEmbedding()
    )
    """
    if args.peft_type == 'lora':
        if args.init_lora_weights == 'none':
            init_lora_weights = True
        else:
            init_lora_weights = args.init_lora_weights
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
            lora_dropout=args.lora_dropout
        )
    elif args.peft_type == 'loha':
        lora_config = LoHaConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
            lora_dropout=args.lora_dropout
        )
    elif args.peft_type == 'lokr':
        lora_config = LoKrConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
            lora_dropout=args.lora_dropout
        )

    if args.peft_type != 'none':
        model.config.tie_word_embeddings = model.config.weight_tying # LoRA expects `tie_word_embeddings`
        if not hasattr(model.config, "get"):
            model.config.get = lambda key, default=None: getattr(model.config, key, default)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to('cpu')
    scripted_model = torch.jit.script(model)
    scripted_model.save("jit_omodel.pt")




if __name__ == "__main__":
    args = get_args()
    main(args)
