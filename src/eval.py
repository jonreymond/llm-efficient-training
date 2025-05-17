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

from datasets import load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

def process(example):
    question = example['Problem']
    choices = {d.strip()[0]:d.split(")")[-1].strip() for d in example['options'].split(",")}
    answer = choices.get(example['correct'])
    
    return {'question': question, 'answer': answer}

def format_qa_pair(q, a):
    return f"Q: {q}\nA: {a}"

def build_few_shot_prompt(data_train, query_sample, n_shots=5):
    # Select 5 random in-context examples that are not the current sample
    few_shot_examples = random.sample([ex for ex in data_train if ex != query_sample], n_shots)
    context = "\n".join([format_qa_pair(ex["question"], ex["answer"]) for ex in few_shot_examples])
    
    final_prompt = f"{context}\nQ: {query_sample['question']}\nA:"
    return final_prompt
    
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
    
    dataset = load_dataset("allenai/math_qa", trust_remote_code=True)
    data_eval = dataset["test"]
    data_eval = data_eval.map(process, remove_columns=['Problem', 'Rationale', 'options', 'correct', 'annotated_formula', 'linear_formula', 'category'])
    print('A sample from dataset:')
    print(data_eval[0])
            
    print(f"Num validation set: {len(data_eval)}")
    
    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'

    # Load from checkpoint
    print(f"Loading from: {args.use_pretrained}")
    last_ckpt_path = args.use_pretrained
    checkpoint = torch.load(last_ckpt_path)
    model_state_dict = {distributed_backend.translate_model_parameter_name_for_node(k.replace("_orig_mod.", ""))[0]:v for k,v in checkpoint['model'].items()}
    # FIXME checkpoints from compiled model have _orig_mod keyword

    model.load_state_dict(model_state_dict) 
    
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

    model.eval()
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'

    # Storage for results
    results = []
    correct = 0

    for sample in tqdm(data_eval):
        if sample["answer"]:
            prompt = build_few_shot_prompt(data_train=data_eval, query_sample=sample, n_shots=5)
            output = model.generate(prompt, max_new_tokens=16)
            generated_answer = output.split("A:")[-1].strip()
        
            results.append({
                "prompt": prompt,
                "generated_response": generated_answer,
                "true_response": sample["answer"]
            })

            if generated_answer and generated_answer.lower() == sample["answer"].lower():
                correct += 1
    
    # Save all responses
    os.makedirs(args.results_path, exist_ok=True)
    with open(f"{args.results_path}/inference_results.json", "w") as f_out:
        json.dump(results, f_out, indent=2)
    
    # Compute and save accuracy
    accuracy = correct / len(data_eval)
    with open(f"{args.results_path}/accuracy.json", "w") as f_acc:
        json.dump({"accuracy": accuracy}, f_acc, indent=2)
    
if __name__ == "__main__":
    args = get_args()
    main(args)
