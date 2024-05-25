if [ -z "${ablation}" ]; then
    ablation=0
fi
if [ -z "${skip_training}" ]; then
    skip_training=0
fi
if [ -z "${s_init}" ]; then
    s_init="0.01"
fi
if [ -z "${S}" ]; then
    S="0"
fi

if [ $ablation -gt 0 ]; then
    prefix="[A$ablation] ${prefix}"
fi
if [ $skip_training -gt 0 ]; then
    prefix="[BASE] ${prefix}"
fi


if [ "$name" == "Baseline (LoRA)" ]; then
    ft_type=$ft_type
elif [ "$name" == "Baseline (Full FT)" ]; then
    ft_type=2
elif [ "$name" == "DoubleLoRA + BiDirAttn + Tokens" ]; then
    model_type=1
    think_type=5
    think_tokens=2
    pass_type=1
elif [ "$name" == "DoubleLoRA(C) + BiDirAttn" ]; then
    model_type=1
    pass_type=$C
    name="DoubleLoRA(P${C}) + BiDirAttn"
elif [ "$name" == "DoubleLoRA[SK] + BiDirAttn" ]; then
    model_type=1
    pass_type=1
    skip_bidir=$S
    name="DoubleLoRA[SK${S}] + BiDirAttn"
elif [ "$name" == "DoubleLoRA(P4) + BiDirAttn" ]; then
    model_type=1
    pass_type=4
elif [ "$name" == "DoubleLoRA(P3) + BiDirAttn" ]; then
    model_type=1
    pass_type=3
elif [ "$name" == "DoubleLoRA(P101) + BiDirAttn" ]; then
    model_type=1
    pass_type=101
elif [ "$name" == "DoubleLoRA(P8) + BiDirAttn + Tokens" ]; then
    model_type=1
    think_type=5
    think_tokens=2
    pass_type=8
elif [ "$name" == "DoubleLoRA(P9) + BiDirAttn + Tokens" ]; then
    model_type=1
    think_type=5
    think_tokens=2
    pass_type=9
elif [ "$name" == "DoubleLoRA(P9) + BiDirAttn" ]; then
    model_type=1
    pass_type=9
elif [ "$name" == "DoubleLoRA(P10) + BiDirAttn" ]; then
    model_type=1
    pass_type=10
elif [ "$name" == "DoubleLoRA + BiDirAttn + NewTokens" ]; then
    model_type=1
    think_type=5
    think_tokens=2
    pass_type=1
    token_init=108
elif [ "$name" == "DoubleLoRA + BiDirAttn" ]; then
    model_type=1
    pass_type=1
elif [ "$name" == "BiDirAttn + Tokens" ]; then
    model_type=1
    think_type=5
    think_tokens=2
    attn_type=2
elif [ "$name" == "BiDirAttn" ]; then
    model_type=1
    attn_type=2
elif [ "$name" == "Tokens" ]; then
    model_type=1
    think_type=5
    think_tokens=2
else
    echo "Invalid name"
    exit 1
fi



suffix="[${learning_rate}]"
if [ $bf16 -eq 1 ]; then
    suffix="${suffix}[bf16]"
fi
if [ $ft_type -eq 14 ]; then
    suffix="${suffix}[DoRA]"
fi
if [ $ft_type -eq 24 ]; then
    suffix="${suffix}[IA3]"
fi
if [ $rank -ne 8 ]; then
    suffix="${suffix}[R${rank}]"
fi
if [ $loss_masking -eq 0 ]; then
    suffix="${suffix}[NOLM]"
fi
if [ $(echo "$weight_decay > 0.0" | bc) -eq 1 ]; then
    suffix="${suffix}[wd${weight_decay}]"
fi
if [ $(echo "$beta1 != 0.9" | bc) -eq 1 ]; then
    suffix="${suffix}[b1${beta1}]"
fi
if [ $(echo "$beta2 != 0.999" | bc) -eq 1 ]; then
    suffix="${suffix}[b2${beta2}]"
fi
if [ $(echo "$seed != 42" | bc) -eq 1 ]; then
    suffix="${suffix}[S${seed}]"
fi
if [ $(echo "$s_init != 0.01" | bc) -eq 1 ]; then
    suffix="${suffix}[si${s_init}]"
fi

wandb_name="${prefix}${name} ${suffix}"
max_eval_steps=100000
eval_batch_size=1
eval_freq=2500
gradient_checkpointing=0

CUDA_VISIBLE_DEVICES=0 HOME_DIR=$HOME_DIR RUN_ID=$run_id accelerate launch finetune.py \
    --model_type $model_type \
    --think_lr $think_lr \
    --lr_scheduler_type $lr_scheduler_type \
    --weight_decay $weight_decay \
    --attn_type $attn_type \
    --think_type $think_type \
    --ft_type $ft_type \
    --model_path $model_path \
    --gradient_checkpointing $gradient_checkpointing \
    --loss_masking $loss_masking \
    --max_epochs $max_epochs \
    --proj_idx $proj_idx \
    --skip_training $skip_training \
    --ablation $ablation \
    --token_init $token_init \
    --bf16 $bf16 \
    --proj_type $proj_type \
    --n_pass $n_pass \
    --eval_freq $eval_freq \
    --eval_batch_size $eval_batch_size \
    --dataset $dataset \
    --seq_length $seq_length \
    --num_warmup_steps $num_warmup_steps \
    --max_train_steps $max_train_steps \
    --max_eval_steps $max_eval_steps \
    --think_tokens $think_tokens \
    --think_pos $think_pos \
    --learning_rate $learning_rate \
    --s_init $s_init \
    --train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --seed $seed \
    --wandb_name "$wandb_name" \
    --method_name "$name" \
    --wandb_group $group \
    --rank $rank \
    --skip_bidir $skip_bidir \
    --pass_type $pass_type \
    --wandb $wandb \
    --save_dir $save_dir
