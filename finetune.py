import os
import time
import logging
from collections import deque
from tqdm import tqdm

import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW, get_scheduler
import wandb

from dataloaders import create_dataloaders
from args import ThinkArguments, update_config
from models.think_gemma import ThinkGemmaForCausalLM
from models.think_llama import ThinkLlamaForCausalLM
from models.think_phi import ThinkPhiForCausalLM

from peft import (
    IA3Config,
    LoraConfig,
    get_peft_model,
    PeftModelForCausalLM,
    PeftMixedModel,
    prepare_model_for_kbit_training,
)


def log_trainable_params(accelerator, model):
    _trainable_params = 0
    if accelerator.is_main_process:
        print(model)
        # print names and shapes of trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        # print all trainable parameters
        _trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            "trainable params:",
            _trainable_params,
            flush=True,
        )
    return _trainable_params


def get_grouped_params(
    model,
    args,
    no_decay=[
        "bias",
        "ln_1.weight",
        "ln_2.weight",
        "ln_f.weight",
        "norm",
    ],
):
    params_with_wd, params_without_wd, think_embeds = [], [], []
    for n, p in model.named_parameters():
        if "think_embeds" in n:
            think_embeds.append(p)
        elif any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {
            "params": params_with_wd,
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": params_without_wd, "weight_decay": 0.0, "lr": args.learning_rate},
        {
            "params": think_embeds,
            "weight_decay": args.weight_decay,
            "lr": args.think_lr if args.think_lr > 0.0 else args.learning_rate,
        },
    ]


def setup_logging(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.wandb_project,
            vars(args),
            init_kwargs={
                "wandb": {
                    "mode": "online" if args.wandb == 1 else "disabled",
                    "name": args.wandb_name,
                    "group": args.wandb_group,
                }
            },
        )
        run_name = accelerator.trackers[0].run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name


def log_metrics(logger, step, metrics, to_print=True):
    if to_print:
        logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        accelerator.log(metrics, step)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_input_and_labels(batch, args):
    input_ids = batch["input_ids"]
    mask = batch["attention_mask"] == 1
    if args.loss_masking == 1:
        range_matrix = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
            input_ids.shape[0], 1
        )
        _mask = range_matrix >= batch["question_sizes"].unsqueeze(1)
        mask = mask & _mask
    labels = torch.where(mask, input_ids, -100)
    return input_ids, labels


def isCustom(model):
    if isinstance(model, model_arch):
        return True
    if hasattr(model, "module") and isinstance(model.module, model_arch):
        return True
    if (
        isinstance(model, PeftModelForCausalLM) or isinstance(model, PeftMixedModel)
    ) and isinstance(model.base_model.model, model_arch):
        return True

    return False


def get_target_model(model):
    if isinstance(model, PeftModelForCausalLM) or isinstance(model, PeftMixedModel):
        return model.base_model.model
    return model


def evaluate(accelerator, model, args):
    model.eval()
    losses = []
    correct = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        input_ids, labels = get_input_and_labels(batch, args)

        with torch.no_grad():
            if isCustom(model):
                # out = pass_fn(model, tokenizer, input_ids, labels, batch, args)
                out = model(
                    input_ids,
                    prompt_lengths=batch["question_sizes"],
                    labels=labels,
                    use_cache=False,
                    peft_model=model,
                )
            else:
                out = model(input_ids, labels=labels, use_cache=False)
            loss = out.loss

        losses.append(accelerator.gather(loss).mean().item())

        range_matrix = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
            input_ids.shape[0], 1
        )
        mask = range_matrix < batch["question_sizes"].unsqueeze(1)
        pads = batch["attention_mask"]
        mask = mask[:, 1:] | (pads[:, 1:] == 0)

        _correct = torch.where(
            mask[:, -out.logits.shape[1] + 1 :],
            mask[:, -out.logits.shape[1] + 1 :],
            (
                batch["input_ids"][:, -out.logits.shape[1] + 1 :]
                == out.logits.argmax(dim=-1)[:, :-1]
            ),
        ).all(-1)
        correct.append(accelerator.gather(_correct))

        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    correct = torch.cat(correct)
    correct_ratio = (correct.sum() / correct.numel()).item()

    loss = sum(losses) / len(losses)
    try:
        perplexity = torch.exp(torch.tensor(loss)).item()
    except OverflowError:
        perplexity = float("inf")

    return {
        "acc/eval": correct_ratio,
        "loss/eval": loss,
        "perplexity/eval": perplexity,
    }


job_id = os.getenv("RUN_ID", str(int(time.time())))
parser = HfArgumentParser(ThinkArguments)
args = parser.parse_args()
if os.getenv("WANDB") == "0":
    args.wandb = 0

if "gemma" in args.model_path.lower():
    model_arch = ThinkGemmaForCausalLM
elif "llama" in args.model_path.lower():
    model_arch = ThinkLlamaForCausalLM
elif "phi" in args.model_path.lower():
    model_arch = ThinkPhiForCausalLM
else:
    raise ValueError()

accelerator = Accelerator(log_with=["wandb"])
samples_per_step = accelerator.state.num_processes * args.train_batch_size

logger, run_name = setup_logging(args)
logger.info(accelerator.state)

if accelerator.is_main_process:
    wandb.config.update({"job_id": job_id})

model_config = AutoConfig.from_pretrained(args.model_path)
update_config(model_config, args)
if accelerator.is_main_process:
    print(model_config, flush=True)

set_seed(args.seed)
_kwargs = {}
if args.bf16:
    _kwargs["torch_dtype"] = torch.bfloat16
if args.model_type == 0:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=args.ft_type == 5, **_kwargs
    )
elif args.model_type == 1:
    model = model_arch.from_pretrained(
        args.model_path, config=model_config, load_in_8bit=args.ft_type == 5, **_kwargs
    )
    if args.token_init > 0:
        model.model.think_embeds.weight.data = model.model.embed_tokens.weight.data[
            args.token_init : args.token_init + 1
        ].expand(model.model.think_embeds.weight.data.shape[0], -1)
else:
    raise ValueError(f"Unknown model type: {args.model_type}")
if "princeton-nlp" in args.model_path or "huggyllama" in args.model_path:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.architectures[0] = model.__class__.__name__

lora_config = None
if args.ft_type == 4 or args.ft_type == 5 or args.ft_type == 14 or args.ft_type == 24:
    modules = []
    modules_to_save = []
    for name, module in model.named_modules():
        if "_proj" in name and "think_embeds" not in name:
            modules.append(name)
        if "think_embeds" in name or "pass_scale" in name:
            modules_to_save.append(name)
    _kwargs = {}
    if args.ft_type == 14:
        _kwargs["use_dora"] = True
    if args.ft_type == 24:
        lora_config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=modules,
            feedforward_modules=[m for m in modules if "mlp" in m],
            modules_to_save=modules_to_save,
            **_kwargs,
        )
    else:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=1,
            target_modules=modules,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
            **_kwargs,
        )
    if args.ft_type == 5:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.pass_type == 0:
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModelForCausalLM(model, lora_config)
        model.add_adapter("prefill", lora_config)
else:
    for name, p in model.named_parameters():
        p.requires_grad = True
        if args.ft_type == 1:
            if "think_embeds" not in name:
                p.requires_grad = False
        elif args.ft_type == 2:
            if (
                "_proj" not in name
                and "think_embeds" not in name
                and "pass_scale" not in name
            ):
                p.requires_grad = False
        elif args.ft_type == 3:
            if "embed" not in name:
                p.requires_grad = False

if args.gradient_checkpointing == 1:
    model.gradient_checkpointing_enable()

_trainable_params = log_trainable_params(accelerator, model)

set_seed(args.seed)
t_start_0 = time.time()
train_dataloader, eval_dataloader = create_dataloaders(tokenizer, args)
print(f"Creating dataloaders took {time.time() - t_start_0} seconds", flush=True)

optimizer = AdamW(get_grouped_params(model, args), betas=(args.beta1, args.beta2))
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)
accelerator.register_for_checkpointing(lr_scheduler)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


set_seed(args.seed)
model.train()
completed_steps = 0
t_start = time.time()
t_start_0 = t_start
loss_tracking = 0
running_loss = deque(maxlen=args.running_loss_window)

step = 1
_epoch = 1
while True:
    if args.skip_training == 1:
        break
    if _epoch > args.max_epochs:
        break
    for batch in train_dataloader:
        input_ids, labels = get_input_and_labels(batch, args)

        if step < 5 and accelerator.is_main_process:
            print("sample:", tokenizer.decode(input_ids[0]), flush=True)
            print("labels:", labels[0], flush=True)

        if step == 1 and accelerator.is_main_process:
            print(f"Seq. length: {input_ids.shape[-1]}", flush=True)

        if isCustom(model):
            # out = pass_fn(model, tokenizer, input_ids, labels, batch, args)
            out = model(
                input_ids,
                prompt_lengths=batch["question_sizes"],
                labels=labels,
                use_cache=False,
                peft_model=model,
            )
        else:
            out = model(input_ids, labels=labels, use_cache=False)
        raw_loss = out.loss

        avg_loss = (
            accelerator.gather(raw_loss.repeat(args.train_batch_size)).mean().item()
            / args.gradient_accumulation_steps
        )
        loss_tracking += avg_loss
        running_loss.append(avg_loss)
        loss = raw_loss / args.gradient_accumulation_steps

        metrics = {
            "samples": step * samples_per_step,
            "epoch": _epoch,
            "running_loss/train": sum(running_loss) / len(running_loss),
            "loss_per_step/train": loss.item(),
            "steps": completed_steps,
            "trainable_params": _trainable_params,
        }
        if hasattr(model, "pass_scale_k"):
            for n, p in model.pass_scale_k.named_parameters():
                if "default" in n:
                    metrics[f"k_{n}/min"] = p.abs().min().item()
                    metrics[f"k_{n}/mean"] = p.abs().mean().item()
                    metrics[f"k_{n}/max"] = p.abs().max().item()
            for n, p in model.pass_scale_v.named_parameters():
                if "default" in n:
                    metrics[f"v_{n}/min"] = p.abs().min().item()
                    metrics[f"v_{n}/mean"] = p.abs().mean().item()
                    metrics[f"v_{n}/max"] = p.abs().max().item()

        if step % args.gradient_accumulation_steps != 0:
            # Prevent backward from doing gradient all_reduce in every step
            if accelerator.distributed_type == DistributedType.MULTI_GPU:
                with model.no_sync():
                    accelerator.backward(loss)
            else:
                accelerator.backward(loss)
        else:
            lr = get_lr(optimizer)
            accelerator.backward(loss)
            if args.clip > 0.0:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.clip)
            else:
                grad_norm = 0.0

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if args.pass_type == 200:
                _ratio = step / (
                    args.num_warmup_steps * args.gradient_accumulation_steps
                )
                _ratio = min(1.0, _ratio)
                for n, p in model.named_parameters():
                    if "pass_scale" in n:
                        p.data.fill_(_ratio)
            elif args.pass_type == 201:
                for n, p in model.named_parameters():
                    if "pass_scale" in n:
                        p.data.fill_(0.5)

            elapsed_time = time.time() - t_start
            t_start = time.time()
            loss_tracking = 0
            completed_steps += 1

            metrics["grad_norm"] = grad_norm
            metrics["lr"] = lr
            metrics["time_per_iteration"] = elapsed_time

        log_metrics(
            logger,
            step,
            metrics,
            to_print=(step - 1) % 10 == 0,
        )

        if step % args.eval_freq == 0:
            logger.info("Evaluating model checkpoint...")
            metrics = evaluate(accelerator, model, args)
            log_metrics(logger, step, metrics)
            accelerator.wait_for_everyone()
            model.train()

        if completed_steps >= args.max_train_steps:
            break

        step += 1
    if completed_steps >= args.max_train_steps:
        break
    _epoch += 1

_training_time = time.time() - t_start_0
print(f"Training took {_training_time} seconds")

logger.info("Evaluating model after training...")
metrics = evaluate(accelerator, model, args)
log_metrics(logger, step, metrics)

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if args.save_dir is not None and args.save_dir != "none":
    logger.info(f"Saving model to {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    unwrapped_model.save_pretrained(args.save_dir, save_function=accelerator.save)
    model.config.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


prompts = []
template = lambda x: f"Question: {x}\n\nAnswer:"

unwrapped_model.eval()
for prompt in prompts:
    try:
        print("============================")
        input_ids = tokenizer.encode(template(prompt), return_tensors="pt")
        input_ids = input_ids.to(accelerator.device)

        kwargs = {}
        if args.model_type == 1:
            kwargs["prompt_lengths"] = torch.tensor(
                [input_ids.shape[1]], device=accelerator.device
            )
            kwargs["peft_model"] = unwrapped_model

        output = unwrapped_model.generate(
            input_ids=input_ids,
            use_cache=True,
            max_length=256,
            **kwargs,
        )

        output = tokenizer.decode(output[0])
        print(output)
    except Exception as e:
        print(e)
