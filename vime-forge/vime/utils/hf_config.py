"""Hugging Face checkpoint configuration loading helpers."""

import json
import os

import torch


def load_hf_config(checkpoint_path):
    """Load HF config, falling back to local metadata for unknown architectures."""
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    except (ValueError, KeyError):
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

        def fix_dtype(config):
            if "torch_dtype" in config:
                config["torch_dtype"] = dtype_map.get(config["torch_dtype"], config["torch_dtype"])
            if "dtype" in config:
                config["dtype"] = dtype_map.get(config["dtype"], config["dtype"])

        fix_dtype(config_dict)
        config = type("HFConfig", (), config_dict)()
        if "text_config" in config_dict:
            fix_dtype(config_dict["text_config"])
            config.text_config = type("TextConfig", (), config_dict["text_config"])()
        return config
