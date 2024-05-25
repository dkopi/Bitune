import os
import datasets
from torch.utils.data import DataLoader


def prepare_datasets(
    args,
    dataset_name,
    tokenizer,
    q_func,
    ans_func,
    train_field="train",
    val_field=None,
    val_subset=None,
    subset=None,
    custom_filter=None,
):
    datasets.disable_caching()
    ds = datasets.load_dataset(dataset_name, subset)

    def tokenize(samples, args):
        samples = [dict(zip(samples, i)) for i in zip(*samples.values())]
        _questions = []
        _full = []
        for _, sample in enumerate(samples):
            q = q_func(sample)
            ans = ans_func(sample)
            _questions.append(q)
            _full.append(q + ans)
        questions = tokenizer(
            _questions,
            padding="max_length" if args.train_batch_size > 1 else "do_not_pad",
            truncation=True if args.seq_length > 0 else False,
            max_length=args.seq_length if args.seq_length > 0 else None,
        )
        full_labels = tokenizer(
            _full,
            padding="max_length" if args.train_batch_size > 1 else "do_not_pad",
            truncation=True if args.seq_length > 0 else False,
            max_length=args.seq_length if args.seq_length > 0 else None,
        )
        question_sizes = [
            len([_q for _q in q if _q != tokenizer.pad_token_id])
            for q in questions["input_ids"]
        ]
        return {
            "input_ids": full_labels["input_ids"],
            "attention_mask": full_labels["attention_mask"],
            "question_sizes": question_sizes,
        }

    train_dataset = ds[train_field]
    if custom_filter is not None:
        train_dataset = custom_filter(train_dataset)
    train_dataset = train_dataset.map(
        lambda samples: tokenize(samples, args),
        remove_columns=train_dataset.column_names,
        batched=True,
        num_proc=24,
    )
    print("before filtering (trainset):", len(train_dataset))
    train_dataset = train_dataset.filter(
        lambda samples: [
            ids[-1] == tokenizer.eos_token_id or ids[-1] == tokenizer.pad_token_id
            for ids in samples["input_ids"]
        ],
        batched=True,
        num_proc=24,
    )
    train_dataset = train_dataset.with_format("torch")
    print("after filtering (trainset):", len(train_dataset))

    if val_subset is not None:
        valid_dataset = train_dataset.select(
            range(len(train_dataset) - val_subset, len(train_dataset))
        )
        train_dataset = train_dataset.select(range(len(train_dataset) - val_subset))
        return train_dataset, valid_dataset

    if val_field is None:
        return train_dataset

    valid_dataset = ds[val_field].map(
        lambda samples: tokenize(samples, args),
        remove_columns=ds[val_field].column_names,
        batched=True,
        num_proc=24,
    )
    print("before filtering (trainset):", len(valid_dataset))
    valid_dataset = valid_dataset.filter(
        lambda samples: [
            ids[-1] == tokenizer.eos_token_id or ids[-1] == tokenizer.pad_token_id
            for ids in samples["input_ids"]
        ],
        batched=True,
        num_proc=24,
    )
    valid_dataset = valid_dataset.with_format("torch")
    print("after filtering (validset):", len(valid_dataset))

    return train_dataset, valid_dataset


prompt_input = (
    lambda x: f"<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n"
)
prompt_output = lambda x: f"{x}<end_of_turn>"


def create_dataloaders(tokenizer, args):
    datasets.disable_caching()

    _path = f"./cached_datasets/{args.model_path}_{args.dataset}_{args.seq_length}_{args.train_batch_size}_{args.seed}"
    if os.path.exists(_path + "_train") and os.path.exists(_path + "_valid"):
        train_dataset = datasets.load_from_disk(_path + "_train")
        valid_dataset = datasets.load_from_disk(_path + "_valid")
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        eval_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size)
        print(f"train size: {len(train_dataset)}, eval size: {len(valid_dataset)}")
        return train_dataloader, eval_dataloader

    if args.dataset == "csqa_full":
        train_dataset, valid_dataset = prepare_datasets(
            args,
            "tau/commonsense_qa",
            tokenizer,
            lambda x: f"Question: {x['question']}{chr(10)}{chr(10)}Choices:{chr(10)}{chr(10).join(x['choices']['text'])}{chr(10)}{chr(10)}Answer:",
            lambda x: " "
            + x["choices"]["text"][x["choices"]["label"].index(x["answerKey"])]
            + tokenizer.eos_token,
            train_field="train",
            val_field="validation",
        )
    elif args.dataset == "arc_full":
        train_dataset, valid_dataset = prepare_datasets(
            args,
            "allenai/ai2_arc",
            tokenizer,
            lambda x: f"Question: {x['question']}{chr(10)}{chr(10)}Choices:{chr(10)}{chr(10).join(x['choices']['text'])}{chr(10)}{chr(10)}Answer:",
            lambda x: " "
            + x["choices"]["text"][x["choices"]["label"].index(x["answerKey"])]
            + tokenizer.eos_token,
            train_field="train",
            val_field="validation",
            subset="ARC-Challenge",
        )
    elif args.dataset == "piqa_full":
        train_dataset, valid_dataset = prepare_datasets(
            args,
            "piqa",
            tokenizer,
            lambda x: f"Question: {x['goal']}{chr(10)}{chr(10)}Choices:{chr(10)}{x['sol1']}{chr(10)}{x['sol2']}{chr(10)}{chr(10)}Answer:",
            lambda x: " "
            + (x["sol1"] if x["label"] == 0 else x["sol2"])
            + tokenizer.eos_token,
            train_field="train",
            val_field="validation",
        )
    elif args.dataset == "siqa_full":
        train_dataset, valid_dataset = prepare_datasets(
            args,
            "social_i_qa",
            tokenizer,
            lambda x: f"Question: Given the context, answer correctly the question.{chr(10)}Context: {x['context']}{chr(10)}Question: {x['question']}{chr(10)}{chr(10)}Choices:{chr(10)}(0) {x['answerA']}{chr(10)}(1) {x['answerB']}{chr(10)}(2) {x['answerC']}{chr(10)}{chr(10)}Answer:",
            lambda x: " " + f"({int(x['label']) - 1})" + tokenizer.eos_token,
            train_field="train",
            val_field="validation",
        )
    elif args.dataset == "openhermes":

        def _prompt(x):
            _input = x["instruction"] + (
                f'{chr(10)}{x["input"]}' if x["input"] != "" else ""
            )
            return f"Question: {_input}{chr(10)}{chr(10)}Answer:"

        train_dataset, valid_dataset = prepare_datasets(
            args,
            "teknium/openhermes",
            tokenizer,
            _prompt,
            lambda x: " " + x["output"] + tokenizer.eos_token,
            train_field="train",
            val_subset=1000,
        )
    elif args.dataset == "alpaca":

        def _prompt(x):
            _input = x["instruction"] + (
                f'{chr(10)}{x["input"]}' if x["input"] != "" else ""
            )
            return f"Question: {_input}{chr(10)}{chr(10)}Answer:"

        train_dataset, valid_dataset = prepare_datasets(
            args,
            "yahma/alpaca-cleaned",
            tokenizer,
            _prompt,
            lambda x: " " + x["output"] + tokenizer.eos_token,
            train_field="train",
            val_subset=1000,
        )
    elif args.dataset == "ultrafeedback":

        def _filter(ds):
            ds = ds.map(
                lambda x: {
                    "completions": [
                        c for c in x["completions"] if c["model"] == "gpt-4"
                    ]
                }
            )
            return ds.filter(lambda x: len(x["completions"]) > 0)

        train_dataset, valid_dataset = prepare_datasets(
            args,
            "openbmb/UltraFeedback",
            tokenizer,
            lambda x: f"Question: {x['instruction']}{chr(10)}{chr(10)}Answer:",
            lambda x: " " + x["completions"][0]["response"] + tokenizer.eos_token,
            train_field="train",
            val_subset=1000,
            custom_filter=_filter,
        )
    elif args.dataset == "gsm8k":
        train_dataset, valid_dataset = prepare_datasets(
            args,
            "gsm8k",
            tokenizer,
            lambda x: f"Question: {x['question']}\n\nAnswer:",
            lambda x: " " + x["answer"] + tokenizer.eos_token,
            subset="main",
            train_field="train",
            val_field="test",
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataset.save_to_disk(_path + "_train")
    valid_dataset.save_to_disk(_path + "_valid")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader
