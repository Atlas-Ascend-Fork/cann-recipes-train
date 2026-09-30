import sys

from vime.utils import accelerator

if accelerator.is_available() and accelerator.device_type() == "npu":
    # torchtitan_npu first: it loads torch_npu._inductor, which calls get_gpu_type.cache_clear(),
    # and transfer_to_npu replaces that function with an uncached one.
    import torchtitan_npu  # noqa: F401
    from torch._dynamo import trace_rules
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    # transfer_to_npu points the torch.cuda seed names at torch_npu's in-graph functions, which
    # torch's own table marks as skipped; dynamo refuses a rule map holding one object under two
    # rules and raises on anything that reaches it. Dropping the skip entries leaves torch_npu's.
    _required = ("torch.cuda.random.manual_seed", "torch.cuda.random.manual_seed_all")
    _dropped = set()
    for _name in _required + ("torch.cuda.manual_seed", "torch.cuda.manual_seed_all"):
        if trace_rules.manual_torch_name_rule_map.pop(_name, None) is not None:
            _dropped.add(_name)

    _missing = [_name for _name in _required if _name not in _dropped]
    if _missing:
        import torch
        import torch_npu

        torch_version = getattr(torch, "__version__", "unknown")
        torch_npu_version = getattr(torch_npu, "__version__", "unknown")
        torch_npu_path = getattr(torch_npu, "__file__", "unknown")
        raise RuntimeError(
            "Incompatible torch/torch_npu Dynamo rule tables: "
            f"the required entries {_missing} are absent after torch_npu's transfer_to_npu "
            "patching. This usually means the torch and torch_npu builds are not the "
            "supported pair for this Torchtitan-NPU checkout. "
            f"Detected python={sys.executable}, torch={torch_version}, "
            f"torch_npu={torch_npu_version} ({torch_npu_path}). "
            "For this checkout, use the matching torch_npu 2.14.0 environment."
        )
