### Tokenizer

        self.tokenizer = tiktoken.get_encoding("gpt2")


- vocab size = 50257

- want to increase to 50304 (nearest multiple of 64) => already default vocab_size (good job!)




### Runtimes

### Base model (number of parameters: 123.59M)

Time per iter (batch_size:32, grad_acc:4):

vanilla: 424 ms

compiled model: 380 ms (11% gain)

compiled model (batch size 64, grad_acc:4): 730.54 ms

### Llama2 (number of parameters: 123.98M)

 Time per iter:

vanilla: 555.99 ms

#### Noam (number of parameters: 109.82M, n_layer=10)

Time per iter:

vanilla: 495 ms

rope fixed + compiled: 440 ms (12.5 % speedup)

rope fixed + compiled (n_layer=12): 495 ms 


### Compute-optimal

21:1 tokens to parameter count

approximately 1.6B tokens processed in 3 hours for vanilla version => 76M params


compiled => 76* 1.11 = 84M params



### Running

noam_large.yaml [running on sandbox2] https://wandb.ai/entropyy/lauzhack-llm/runs/9uyztfyt/workspace?nw=nwuserentropyy

noam_small [sandbox]


### Best config yet

https://wandb.ai/entropyy/lauzhack-llm/runs/ljxg01g5?nw=nwuserentropyy - noam.yaml - final-acc: 0.4343  - final-loss: 3.134 -  final-pp: 22.96 


(New Contender)

new_xl_noam_lr_0.001_bs64x4  - final-acc: 0.434 - final-loss: 3.125 - final-pp: 22.763
https://wandb.ai/entropyy/lauzhack-llm/runs/t3oeb4wr?nw=nwuserentropyy


(New Contender)

new2_xl_noam_lr_0.001_bs64x4 - final-acc: 0.4344 - final-loss: 3.123 - final-pp: 22.725
https://wandb.ai/entropyy/lauzhack-llm/runs/dynbcdcw?nw=nwuserentropyy


(Newest Contender)

(230M parameters) new2_wide_noam_lr_0.001_bs64x4 - final-acc: 0.4347 - final-loss: 3.112 - final-pp: 22.473
https://wandb.ai/entropyy/lauzhack-llm/runs/pijawqpn?nw=nwuserentropyy


```
model: "noam"
weight_tying: True

compile: True
grad_clip: 1.0
save_checkpoint_freq: 500
wandb_project: "lauzhack-llm"
wandb: True
n_layer: 10
opt: "adamw"
weight_decay: 0.1
lr: 1.e-3
#rho: 0.05
run_prefix: "noam_"

```


### things to tune

- lr
- warmup




### Defaults


good:



- 5% percent warmup
- weight_decay: 0.1

- bias: False
- multiple_of: 256

to tweak:


- batch size: 32
- acc_grad_steps: 4
- iterations: 25'000
- lr: 1e-3

- beta1: 0.9
- beta2: 0.95

- scheduler: cos
- opt: adamw

- eval_freq: 100

-  grad_clip: 0.0 => 1.0

- dropout: 0.0 => ?

- seq_length: 512 => ? 

- save_checkpoint_freq: None => 100 (NEED TO CHANGE)



