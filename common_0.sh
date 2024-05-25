HOME_DIR="/path/to/this/dir"


run_id=420 # any value here
wandb=1
save_dir="ckpt/${run_id}/"

batch_size=1

lr_scheduler_type="linear"
gradient_accumulation_steps=10
seq_length=512
max_epochs=100

seeds=(42 43 44)
seed=${seeds[$CUSTOM_IDX]}

model_type=0
think_type=0
think_tokens=0
think_pos=0
attn_type=0
ft_type=4
pass_type=0
proj_type=0
proj_idx=0
n_pass=0
token_init=0
skip_bidir=0

bf16=1
rank=8
loss_masking=1
think_lr="0.0"
weight_decay=0.0
beta1=0.9
beta2=0.999
