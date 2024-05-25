import os
import json
import torch
from transformers import HfArgumentParser
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from accelerate import Accelerator
import wandb
import time

from argparse import ArgumentParser


def eval(model, tasks, bs=64, limit=None, num_fewshot=0):
    out = simple_evaluate(  # call simple_evaluate
        model=model,
        tasks=tasks,
        batch_size=bs,
        device="cuda",
        limit=limit,
        num_fewshot=num_fewshot,
    )
    results = out["results"]

    metrics = {}
    for task in results:
        for metric_raw in results[task]:
            _split = metric_raw.split(",")
            if len(_split) == 1:
                metric = metric_raw
            else:
                metric = f'{metric_raw.split(",")[1]}/{metric_raw.split(",")[0]}'
            if metric == "alias":
                continue
            value = results[task][metric_raw]
            metrics[f"{task}/{metric}"] = value

    return metrics, out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=int, required=True)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--update", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--remove", type=int, default=0)
    args = parser.parse_args()

    if args.update == 1:
        time.sleep(10)
        api = wandb.Api()
        runs = api.runs("username/bitune", {"group": args.group}, per_page=10000)

        run = None
        for r in runs:
            if "job_id" in r.config and r.config["job_id"] == str(args.model_id):
                run = r
                break

        if run is None:
            print(f"Run with job_id {args.model_id} not found")
            exit(0)

    home_dir = os.getenv("HOME_DIR")
    pretrained = f"{home_dir}/ckpt/{args.model_id}"

    remove = False
    if os.path.exists(pretrained + "/adapter_config.json"):
        with open(pretrained + "/adapter_config.json", "r") as f:
            base_model_name_or_path = json.load(f)["base_model_name_or_path"]
        model = HFLM(pretrained=pretrained, peft=pretrained)
    else:
        model = HFLM(pretrained=pretrained)
        remove = True if args.remove == 1 else False

    metrics, out = eval(
        model, [args.task], bs=args.bs, limit=args.limit, num_fewshot=args.num_fewshot
    )

    print(f"Results for model {args.model_id}:")

    for i in range(5):
        print("===================")
        try:
            _sample = next(iter(out["samples"].values()))
            print(_sample[i]["arguments"][0][0])
            print(_sample[i]["resps"][0][0])
        except Exception as e:
            print(e)

    for metric in metrics:
        if args.num_fewshot > 0:
            metric = f"{metric}({args.num_fewshot}-shot)"
        if args.update == 1:
            run.summary[metric] = metrics[metric]
        print(f"{metric}: {metrics[metric]}", flush=True)

    if args.update == 1:
        run.update()
        time.sleep(10)
        run.update()
        if remove:
            os.system(f"rm -rf {pretrained}")

    if "KEK" in os.environ:
        breakpoint()
