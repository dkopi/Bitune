<h1 align="center"> 
    <img src="./imgs/header.jpg" width="600">
</h1>
<h2 align="center">
    <p>Bitune: Bidirectional Instruction-Tuning</p>
</h2>

[[`Paper`](https://arxiv.org/abs/2405.14862)] [[`Website`](https://dkopi.github.io/bitune/)]

This source code contains the implementation of Bitune, and it's sufficient to reproduce the results from the paper. Please note that it was used to explore different ideas, and many components have different names or refer to concepts not mentioned in the paper.

**We plan to release a clean repo for Bitune in the near future.**

## lm-evaluation-harness

The `lm-evaluation-harness` directory contains the repository from [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), adapted to our method. You can install it with the following command:

```bash
pip install -e lm-evaluation-harness
```

## Configuration

- Set the proper absolute path to this directory in the `common_0.sh` file.
- The evaluation script requires `wandb` for logging. Update line 57 of `eval.py` with your `wandb` username.

## Scripts

- **Instruction-Tuning Setup**: Run the `instruct.sh` script.
- **Downstream Task Training**: Run the `downstream.sh` script. Ensure to set the correct number of update steps (based on the values provided in the appendix), and uncomment the appropriate lines for the dataset name, evaluations (at the very bottom), and the method name.

## A Brief Overview of the <span style="font-size:12px;text-decoration: line-through;">Spaghetti</span> Code

- Implementation required a few modifications of HuggingFace model classes, available in the `models` directory:
  - Modified KV-cache, so it keeps the computation graph for gradients.
  - Added mixing modules with trainable coefficients (`pass_scale_k`, `pass_scale_v`).
  - Modified attention mask based on `enforce_bidir` parameter of the `forward()` function.
  - Added a code snippet inside the `forward()` function responsible for calling the _Bitune wrapper_.
- The _Bitune wrapper_ (`_pass_fn()` in the `passes.py` file):
  - Passes the prompt through the model two times two obtain to sets of KV-cache, while setting proper LoRA adapters & attention masks for each pass.
  - Calls mixing modules to combine two sets of features (`pass_scale_k`, `pass_scale_v`).
  - Sets all LoRA's parameters as trainable again, as by default `peft` library sets inactive adapters as non-trainable.
- The mixing module (class `PassScale` defined in `models/think_gemma.py`):
  - Contains trainable coefficients for mixing two sets of features.
  - Defines `forward()` function that applies the mixing operation based on the variant specified in the config (`config.pass_type`). Our final method is defined by the variant `607` (the one used for experiments), and it's simplified version `801`.

## Library Versions

The following versions of the libraries have been used:

- `transformers==4.38.2`
- `peft==0.11.1`
- `datasets==2.18.0`
- `evaluate==0.4.0`

## Bibtex

```bibtex
@misc{kopiczko2024vera,
    title={VeRA: Vector-based Random Matrix Adaptation},
    author={Dawid J. Kopiczko and Tijmen Blankevoort and Yuki M. Asano},
    year={2024},
    eprint={2310.11454},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
