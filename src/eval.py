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
from data.utils import get_dataset, get_dataloader
from optim.base import train_base
from optim.utils import eval
import distributed
import itertools
from contextlib import nullcontext


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

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
    if args.data_in_ram:
        data = {'val': np.array(data['val'])}
        
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'

    model = distributed_backend.transform_model(model)
    
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of parameters: %.2fM" % (optimized_params_cnt/1e6,))
   
    
    args.world_size = distributed_backend.get_world_size()
    exp_name = args.exp_name

    
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
        distributed_backend.sync()



    if args.use_pretrained == "auto":
        checkpoints = [file for file in os.listdir(ckpt_path) if 'ckpt_' in file]
        if checkpoints:
            args.use_pretrained = sorted(checkpoints)[-1]
        else:
            args.use_pretrained = None
    
    if args.use_pretrained is not None:
        last_ckpt_path = args.use_pretrained
        checkpoint = torch.load(os.path.join(ckpt_path, last_ckpt_path))
        print(f"Resuming from {os.path.join(ckpt_path, last_ckpt_path)}")
        model_state_dict = {distributed_backend.translate_model_parameter_name_for_node(k.replace("_orig_mod.", ""))[0]:v for k,v in checkpoint['model'].items()}
        # FIXME checkpoints from compiled model have _orig_mod keyword

        model.load_state_dict(model_state_dict) 


    print(f"\Evaluating model={args.model} \n{vars(args)}\n")

    model.eval()


    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16) 


    data_val_iter, val_sampler = get_dataloader(
        data["val"],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.data_seed,
    )

    eval_steps = len(data_val_iter)

    data_val_iter = itertools.cycle(data_val_iter)




    val_acc, val_loss, val_perplexity = eval(
        model,
        data_val_iter,
        args.device,
        max_num_batches=eval_steps,
        ctx=type_ctx,
    )

    print(f"[val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}")


    logs = {
        "val/loss": val_loss,
        "val/perplexity": val_perplexity,
        "val/acc": val_acc,
    }


    
    args.device = None
    args.dtype = None
    logs['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary_val.json", "w") as fs:
            json.dump(logs, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
