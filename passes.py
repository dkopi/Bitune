import torch
import os


def enable_loras(model):
    # enable gradient for all params containing "lora"
    for name, param in model.named_parameters():
        if "lora" in name or "think_embeds" in name or "pass_scale" in name or "ia3_l" in name:
            param.requires_grad = True


def prepare_model_for_pass(model, lora_config, args):
    # if args.pass_type == 13 or args.pass_type == 14:
    #     model.add_adapter("prefill0", lora_config)
    #     model.set_adapter(["default", "prefill", "prefill0"])
    # if args.pass_type == 5:
    #     model.add_adapter("prefill2", lora_config)
    #     model.set_adapter(["default", "prefill", "prefill2"])
    # if args.pass_type == 7 or args.pass_type == 8 or args.pass_type == 10:
    #     model.add_adapter("prefill0", lora_config)
    #     model.add_adapter("prefill1", lora_config)
    #     model.set_adapter(["default", "prefill", "prefill0", "prefill0"])

    # model_config2 = AutoConfig.from_pretrained("google/gemma-7b")
    # update_config(model_config2, args)
    # model2 = ThinkGemmaForCausalLM.from_pretrained("google/gemma-7b", config=model_config2)
    # for name, module in model2.named_modules():
    #     if "_proj" in name and "think_embeds" not in name:
    #         modules.append(name)
    #     if "think_embeds" in name:
    #         modules_to_save.append(name)
    # config2 = LoraConfig(
    #     r=args.rank,
    #     lora_alpha=1,
    #     target_modules=modules,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model2 = prepare_model_for_kbit_training(model2, use_gradient_checkpointing=False)
    # model2 = get_peft_model(model2, config2)
    # model.bigone = model2

    # model.kek1 = torch.nn.Linear(256, 256)
    # model.kek2 = torch.nn.Linear(256, 256)
    # model.kekproj = torch.nn.Linear(256 * 16, 256)
    # model.kekproj.weight.data.zero_()

    return model


def _pass_fn(model, **kwargs):
    if model.config.pass_type == 1 or model.config.pass_type == 2:
        # Q (without last token) + A
        assert "input_ids" in kwargs
        assert "prompt_lengths" in kwargs
        assert kwargs["prompt_lengths"].shape[0] == 1
        enable_loras(model)

        input_ids = kwargs["input_ids"]
        prompt_lengths = kwargs["prompt_lengths"]
        labels = kwargs["labels"] if "labels" in kwargs else None
        use_cache = kwargs["use_cache"] if "use_cache" in kwargs else False

        if model.config.ablation == 2:
            model.set_adapter("default")
        else:
            model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, : prompt_lengths[0] - 1],
            prompt_lengths=prompt_lengths,
            use_cache=True,
            enforce_bidir=model.config.pass_type == 1,
            call_depth=1,
        )["past_key_values"]

        kwargs = {
            "cache_position": torch.arange(
                prompt_lengths[0] - 1 + model.config.think_tokens,
                input_ids.shape[1] + model.config.think_tokens,
                device=input_ids.device,
            ),
        }
        if "phi" in model.config.architectures[0].lower():
            kwargs["position_ids"] = kwargs["cache_position"].unsqueeze(0)
            del kwargs["cache_position"]

        model.set_adapter("default")
        out = model(
            input_ids[:, prompt_lengths[0] - 1 :],
            prompt_lengths=prompt_lengths,
            labels=labels[:, prompt_lengths[0] - 1 :] if labels is not None else None,
            use_cache=use_cache,
            past_key_values=past_key_values,
            call_depth=1,
            **kwargs,
        )
        enable_loras(model)
        return out
    elif model.config.pass_type >= 3:
        assert "input_ids" in kwargs
        assert "prompt_lengths" in kwargs
        assert kwargs["prompt_lengths"].shape[0] == 1
        enable_loras(model)

        input_ids = kwargs["input_ids"]
        prompt_lengths = kwargs["prompt_lengths"]
        labels = kwargs["labels"] if "labels" in kwargs else None
        use_cache = kwargs["use_cache"] if "use_cache" in kwargs else False

        if not hasattr(model.config, "ablation"):
            model.config.ablation = 0

        if model.config.ablation == 2:
            model.set_adapter("default")
        else:
            model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, : prompt_lengths[0] - 1],
            prompt_lengths=prompt_lengths,
            use_cache=True,
            enforce_bidir=False if model.config.ablation == 1 else True,
            call_depth=1,
        )["past_key_values"]

        model.set_adapter("default")
        past_key_values_causal = model(
            input_ids[:, : prompt_lengths[0] - 1],
            prompt_lengths=prompt_lengths,
            use_cache=True,
            enforce_bidir=False,
            call_depth=1,
        )["past_key_values"]

        _past_key_values = ()
        for i, (k, v) in enumerate(past_key_values_causal):
            _k, _v = past_key_values[i]
            _past_key_values += (
                (model.pass_scale_k(k, _k, i), model.pass_scale_v(v, _v, i)),
            )
        past_key_values = _past_key_values

        kwargs = {
            "cache_position": torch.arange(
                prompt_lengths[0] - 1 + model.config.think_tokens,
                input_ids.shape[1] + model.config.think_tokens,
                device=input_ids.device,
            ),
        }
        if "phi" in model.config.architectures[0].lower():
            kwargs["position_ids"] = kwargs["cache_position"].unsqueeze(0)
            del kwargs["cache_position"]

        out = model(
            input_ids[:, prompt_lengths[0] - 1 :],
            prompt_lengths=prompt_lengths,
            labels=labels[:, prompt_lengths[0] - 1 :] if labels is not None else None,
            use_cache=use_cache,
            past_key_values=past_key_values,
            call_depth=1,
            **kwargs,
        )
        enable_loras(model)
        return out
    else:
        raise NotImplementedError(f"Pass type {model.config.pass_type} not implemented")


def pass_fn(model, tokenizer, input_ids, labels, batch, args):
    if args.pass_type == 0:
        out = model(
            input_ids,
            prompt_lengths=batch["question_sizes"],
            labels=labels,
            use_cache=False,
        )
    elif args.pass_type == 10:
        # loops with prefix
        assert batch["question_sizes"].shape[0] == 1
        assert args.think_tokens == 0

        _prefill_len = batch["question_sizes"][0] - 1
        suffix = tokenizer.encode(
            "<hint>correct answer</hint>", return_tensors="pt"
        ).to(input_ids.device)[:, 1:]

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        for i in range(2):
            model.set_adapter("prefill" + str(i))
            _prev_len = past_key_values[0][0].shape[2]
            past_key_values = model(
                suffix,
                prompt_lengths=batch["question_sizes"],
                use_cache=True,
                cache_position=torch.arange(
                    _prev_len,
                    _prev_len + suffix.shape[1],
                    device=input_ids.device,
                ),
                past_key_values=past_key_values,
                enforce_bidir=True,
            )["past_key_values"]

        model.set_adapter("default")
        out = model(
            input_ids[:, _prefill_len:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, _prefill_len:],
            use_cache=False,
            cache_position=torch.arange(
                past_key_values[0][0].shape[2],
                past_key_values[0][0].shape[2] + input_ids[:, _prefill_len:].shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        model.set_adapter(["default", "prefill", "prefill0", "prefill1"])
    elif args.pass_type == 9:
        # loops with whole Qs
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1
        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        _past_key_values = ()
        for i, (k, v) in enumerate(past_key_values):
            if i >= 6:
                _past_key_values += ((k, v),)
            else:
                _past_key_values += ((torch.zeros_like(k), torch.zeros_like(v)),)
                # (torch.zeros_like(k)[:, :, :0], torch.zeros_like(v)[:, :, :0]),
        past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids[:, 1:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, 1:],
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens,
                _prefill_len + args.think_tokens + input_ids.shape[1] - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        model.set_adapter(["default", "prefill"])
    elif args.pass_type == 8:
        # loops with prefix
        assert batch["question_sizes"].shape[0] == 1
        assert args.think_tokens == 0

        _prefill_len = batch["question_sizes"][0]
        prefix = tokenizer.encode("<start>hint<end>", return_tensors="pt").to(
            input_ids.device
        )
        prefixed_prompt = torch.cat([prefix, input_ids[:, 1:_prefill_len]], dim=1)

        model.set_adapter("prefill")
        past_key_values = model(
            prefixed_prompt,
            prompt_lengths=prefixed_prompt.shape[1],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        _past_key_values = ()
        for k, v in past_key_values:
            _past_key_values += (
                (
                    k[:, :, : prefix.shape[1]],
                    v[:, :, : prefix.shape[1]],
                ),
            )
        past_key_values = _past_key_values

        for i in range(2):
            model.set_adapter("prefill" + str(i))
            _prev_len = past_key_values[0][0].shape[2]
            past_key_values = model(
                prefixed_prompt[:, 1:],
                prompt_lengths=prefixed_prompt.shape[1],
                use_cache=True,
                cache_position=torch.arange(
                    _prev_len,
                    _prev_len + prefixed_prompt.shape[1] - 1,
                    device=input_ids.device,
                ),
                past_key_values=past_key_values,
                enforce_bidir=True,
            )["past_key_values"]

            _past_key_values = ()
            for k, v in past_key_values:
                _past_key_values += (
                    (
                        k[:, :, : _prev_len + prefix.shape[1] - 1],
                        v[:, :, : _prev_len + prefix.shape[1] - 1],
                    ),
                )
            past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids[:, 1:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, 1:],
            use_cache=False,
            cache_position=torch.arange(
                past_key_values[0][0].shape[2],
                past_key_values[0][0].shape[2] + input_ids.shape[1] - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        model.set_adapter(["default", "prefill", "prefill0", "prefill0"])
    elif args.pass_type == 7:
        # loops
        assert batch["question_sizes"].shape[0] == 1
        assert args.think_tokens == 0
        _prefill_len = batch["question_sizes"][0]

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
        )["past_key_values"]

        for i in range(2):
            model.set_adapter("prefill" + str(i))
            past_key_values = model(
                input_ids[:, :_prefill_len],
                prompt_lengths=batch["question_sizes"],
                use_cache=True,
                cache_position=torch.arange(
                    _prefill_len,
                    _prefill_len * 2,
                    device=input_ids.device,
                ),
                past_key_values=past_key_values,
            )["past_key_values"]

            _past_key_values = ()
            for k, v in past_key_values:
                _past_key_values += (
                    (
                        k[:, :, -_prefill_len:],
                        v[:, :, -_prefill_len:],
                    ),
                )
            past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids,
            prompt_lengths=batch["question_sizes"],
            labels=labels,
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len,
                _prefill_len + input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        model.set_adapter(["default", "prefill", "prefill0", "prefill0"])
    elif args.pass_type == 1:
        # Q (without last token) + A
        assert batch["question_sizes"].shape[0] == 1
        enable_loras(model)

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, : batch["question_sizes"][0] - 1],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        model.set_adapter("default")
        out = model(
            input_ids[:, batch["question_sizes"][0] - 1 :],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, batch["question_sizes"][0] - 1 :],
            use_cache=False,
            cache_position=torch.arange(
                batch["question_sizes"][0] - 1,
                input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        model.set_adapter("prefill")
        enable_loras(model)
    elif args.pass_type == 69:
        # Q (without last token) + A
        assert batch["question_sizes"].shape[0] == 1

        # with torch.no_grad():
        past_key_values = model.bigone(
            input_ids[:, : batch["question_sizes"][0] - 1],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        _past_key_values = ()
        for i, (k, v) in enumerate(past_key_values):
            _past_key_values += (
                (
                    model.base_model.model.kekproj(
                        k.transpose(1, 2).view(1, -1, 256 * 16)
                    ).view(1, 1, -1, 256),
                    model.base_model.model.kekproj(
                        v.transpose(1, 2).view(1, -1, 256 * 16)
                    ).view(1, 1, -1, 256),
                ),
            )
        past_key_values = _past_key_values[-18:]

        model.set_adapter("default")
        out = model(
            input_ids,
            prompt_lengths=batch["question_sizes"],
            labels=labels,
            use_cache=False,
            cache_position=torch.arange(
                batch["question_sizes"][0] - 1,
                batch["question_sizes"][0] - 1 + input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill"])
    elif args.pass_type == 420:
        # Q (without last token) + A
        assert batch["question_sizes"].shape[0] == 1

        with torch.no_grad():
            model.set_adapter("prefill")
            past_key_values = model(
                input_ids[:, : batch["question_sizes"][0] - 1],
                prompt_lengths=batch["question_sizes"],
                use_cache=True,
                enforce_bidir=True,
            )["past_key_values"]

        _past_key_values = ()
        for i, (k, v) in enumerate(past_key_values):
            _past_key_values += (
                (
                    model.base_model.model.keks[i](k),
                    model.base_model.model.keks[i](v),
                ),
            )
        past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids[:, batch["question_sizes"][0] - 1 :],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, batch["question_sizes"][0] - 1 :],
            use_cache=False,
            cache_position=torch.arange(
                batch["question_sizes"][0] - 1,
                input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill"])
    elif args.pass_type == 11:
        # Q (without last token) + A (singe lora)
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0]

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, : batch["question_sizes"][0] - 1],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        out = model(
            input_ids[:, batch["question_sizes"][0] - 1 :],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, batch["question_sizes"][0] - 1 :],
            use_cache=False,
            cache_position=torch.arange(
                batch["question_sizes"][0] - 1 + args.think_tokens,
                input_ids.shape[1] + args.think_tokens,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
    elif args.pass_type == 12:
        # Q (without last token) + A (singe lora, no bidir)
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0]

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, : batch["question_sizes"][0] - 1],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
        )["past_key_values"]

        out = model(
            input_ids[:, batch["question_sizes"][0] - 1 :],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, batch["question_sizes"][0] - 1 :],
            use_cache=False,
            cache_position=torch.arange(
                batch["question_sizes"][0] - 1 + args.think_tokens,
                input_ids.shape[1] + args.think_tokens,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
    elif args.pass_type == 2:
        # last token of Q + QA
        assert batch["question_sizes"].shape[0] == 1

        # _model = get_target_model(model)
        # _think_type = _model.config.think_type

        # _model.config.think_type = 1
        _prefill_len = batch["question_sizes"][0]
        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]
        # _model.config.think_type = _think_type

        _past_key_values = ()
        for i, (k, v) in enumerate(past_key_values):
            if i >= 0:
                _past_key_values += (
                    (
                        # k[:, :, -1:],
                        # v[:, :, -1:],
                        # k.mean(dim=2, keepdim=True),
                        # v.mean(dim=2, keepdim=True),
                        model.base_model.model.kek1(k.mean(dim=2, keepdim=True)),
                        model.base_model.model.kek2(v.mean(dim=2, keepdim=True)),
                    ),
                )
            else:
                _past_key_values += (
                    (torch.zeros_like(k)[:, :, :0], torch.zeros_like(v)[:, :, :0]),
                )
        past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids,
            prompt_lengths=batch["question_sizes"],
            labels=labels,
            use_cache=False,
            cache_position=torch.arange(
                1,
                1 + input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill"])
    elif args.pass_type == 3:
        # Q + QA(nobos)
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1
        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        # _past_key_values = ()
        # for i, (k, v) in enumerate(past_key_values):
        #     if i >= 6:
        #         _past_key_values += ((k, v),)
        #     else:
        #         _past_key_values += (
        #             (torch.zeros_like(k)[:, :, :0], torch.zeros_like(v)[:, :, :0]),
        #         )
        # past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids[:, 1:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, 1:],
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens,
                _prefill_len + args.think_tokens + input_ids.shape[1] - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill"])
    elif args.pass_type == 14:
        # Q + Q + A(nobos) [3 loras]
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        model.set_adapter("prefill0")
        past_key_values = model(
            input_ids[:, 1:_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
            cache_position=torch.arange(
                _prefill_len,
                _prefill_len + _prefill_len - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )["past_key_values"]

        model.set_adapter("default")
        out = model(
            input_ids[:, _prefill_len:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, _prefill_len:],
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + _prefill_len - 1,
                _prefill_len + input_ids.shape[1] - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill", "prefill0"])
    elif args.pass_type == 33:
        # Q + QA(nobos) [single lora]
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        out = model(
            input_ids[:, 1:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, 1:],
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens,
                _prefill_len + args.think_tokens + input_ids.shape[1] - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
    elif args.pass_type == 13:
        # Q(dropped) + Q(nobos) + A(nobos) [3 loras]
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1

        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            enforce_bidir=True,
        )["past_key_values"]

        model.set_adapter("prefill0")
        past_key_values = model(
            input_ids[:, 1:_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            cache_position=torch.arange(
                _prefill_len,
                _prefill_len + _prefill_len - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
            enforce_bidir=True,
        )["past_key_values"]

        _past_key_values = ()
        for k, v in past_key_values:
            _past_key_values += (
                (
                    torch.cat([k[:, :, :1], k[:, :, _prefill_len:]], dim=2),
                    torch.cat([v[:, :, :1], v[:, :, _prefill_len:]], dim=2),
                ),
            )
        past_key_values = _past_key_values

        model.set_adapter("default")
        out = model(
            input_ids[:, 1:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, 1:],
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len,
                _prefill_len + input_ids.shape[1] - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill", "prefill0"])
    elif args.pass_type == 4:
        # Q + QA
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1
        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
        )["past_key_values"]

        model.set_adapter("default")
        out = model(
            input_ids,
            prompt_lengths=batch["question_sizes"],
            labels=labels,
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens,
                _prefill_len + args.think_tokens + input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill"])
    elif args.pass_type == 5:
        # Q + Q(nobos) + QA(nobos)
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1
        model.set_adapter("prefill")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
        )["past_key_values"]

        model.set_adapter("prefill2")
        past_key_values = model(
            input_ids[:, 1:_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens,
                _prefill_len + args.think_tokens + _prefill_len - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )["past_key_values"]

        model.set_adapter("default")
        out = model(
            input_ids[:, 1:],
            prompt_lengths=batch["question_sizes"],
            labels=labels[:, 1:],
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens + _prefill_len - 1,
                _prefill_len
                + args.think_tokens
                + _prefill_len
                - 1
                + input_ids.shape[1]
                - 1,
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default", "prefill", "prefill2"])
    elif args.pass_type == 6:
        # Q + QA (same adapter)
        assert batch["question_sizes"].shape[0] == 1
        _prefill_len = batch["question_sizes"][0] - 1
        model.set_adapter("default")
        past_key_values = model(
            input_ids[:, :_prefill_len],
            prompt_lengths=batch["question_sizes"],
            use_cache=True,
        )["past_key_values"]

        model.set_adapter("default")
        out = model(
            input_ids,
            prompt_lengths=batch["question_sizes"],
            labels=labels,
            use_cache=False,
            cache_position=torch.arange(
                _prefill_len + args.think_tokens,
                _prefill_len + args.think_tokens + input_ids.shape[1],
                device=input_ids.device,
            ),
            past_key_values=past_key_values,
        )
        # model.set_adapter(["default"])
    else:
        raise ValueError(f"Unknown pass type: {args.pass_type}")
    return out
