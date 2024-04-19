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


### Compute-optimal

21:1 tokens to parameter count

approximately 1.6B tokens processed in 3 hours for vanilla version => 76M params


compiled => 76* 1.11 = 84M params



### Running

base_smallish.yaml [running on sandbox]




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



