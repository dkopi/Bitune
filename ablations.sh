####################
source common_0.sh #
####################

prefix=""
group="instruct"

max_train_steps=3000
num_warmup_steps=$((max_train_steps / 10))

dataset="ultrafeedback"

model_path="google/gemma-2b"
# model_path="meta-llama/Meta-Llama-3-8B"


########## UNCOMMENT THE LINES FOR SELECTED ABLATION ##########

# Full Bitune
C=607
ablation=0

# # Naive Bidir.
# C=1
# ablation=2

# # No Mixing
# C=1
# ablation=0

# # Only Causal
# C=607
# ablation=1

# # Shared Weights
# C=607
# ablation=2

###############################################################


name="DoubleLoRA(C) + BiDirAttn"

for learning_rate in "3e-4"; do


####################
source common_1.sh #
####################


done

HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task piqa_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task arc_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task csqa_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task siqa_full
HOME_DIR=$HOME_DIR python eval.py --group ${group} --model_id ${run_id} --bs 1 --task mmlu --num_fewshot 0
