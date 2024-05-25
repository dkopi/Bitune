####################
source common_0.sh #
####################

prefix=""

max_train_steps=747
num_warmup_steps=$((max_train_steps / 10))

dataset="gsm8k"
# dataset="arc_full"
# dataset="csqa_full"
# dataset="piqa_full"
# dataset="siqa_full"

# model_path="google/gemma-2b"
# model_path="google/gemma-7b"
model_path="meta-llama/Meta-Llama-3-8B"
# model_path="meta-llama/Llama-2-7b-hf"
# model_path="microsoft/phi-2"

group="instruct"

C=607

# name="Baseline (LoRA)" # baseline finetuning
name="DoubleLoRA(C) + BiDirAttn" # Bilora


for learning_rate in "1e-3"; do
# for learning_rate in "3e-4"; do


####################
source common_1.sh #
####################


done

HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task gsm8k --num_fewshot 0
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task arc_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task csqa_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task piqa_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task siqa_full
