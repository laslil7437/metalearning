# Balanced config for NanoGPT on RTX 5000 Ada (32 GB)
out_dir = 'model_ckpts'
eval_interval = 1000
eval_iters = 100
log_interval = 10

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5
batch_size = 16
block_size = 512

# model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

# adamw optimizer
learning_rate = 3e-4
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-5
warmup_iters = 2000

# compute
device = 'cuda'
dtype = 'float16'
compile = False

always_save_checkpoint = True
init_from = 'resume' #  'resume' for stripped, else delete this line to default to 'scratch'

