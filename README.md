# Temporary codebase for Bitune

- [Paper](https://arxiv.org/abs/2405.14862)

This codebase is provided for reproducibility purposes only. Please note that it was used to explore different ideas, and many components have different names than those mentioned in the paper, or refer to new concepts (e.g., Think[...]). We plan to release a clean codebase for Bitune in the future.

### lm-evaluation-harness

The `lm-evaluation-harness` directory contains the repository from [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), adapted to our method. You can install it with the following command:

```bash
pip install -e lm-evaluation-harness
```

### Configuration

- Set the proper absolute path to this directory in the `common_0.sh` file.
- The pipeline requires `wandb` for logging. Update line 57 of `eval.py` with your `wandb` username.

### Scripts

- **Instruction-Tuning Setup**: Run the `instruct.sh` script.
- **Downstream Task Training**: Run the `downstream.sh` script. Ensure to set the correct number of update steps (based on the values provided in the appendix), and uncomment the appropriate lines for the dataset name, evaluations (at the very bottom), and the method name.

## Library Versions

The following versions of the libraries have been used:

- `transformers==4.38.2`
- `peft==0.11.1` (with a custom fix for DoRA, will do a PR later)
- `datasets==2.18.0`
- `evaluate==0.4.0`
