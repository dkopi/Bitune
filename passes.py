import torch
import os


def enable_loras(model):
    # enable gradient for all params containing "lora"
    for name, param in model.named_parameters():
        if (
            "lora" in name
            or "think_embeds" in name
            or "pass_scale" in name
            or "ia3_l" in name
        ):
            param.requires_grad = True


def _pass_fn(model, **kwargs):
    if model.config.pass_type == 1 or model.config.pass_type == 2:
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
