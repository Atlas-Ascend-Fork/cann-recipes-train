"""Training quantization for VIME-owned TorchTitan model construction."""

import logging

from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def apply_training_quantization(config):
    """Apply the NPU quantization recipe before VIME builds the model.

    ``TitanEngine`` assembles the model itself and therefore does not pass
    through ``torchtitan_npu.extensions.TrainerEx.__init__``.  Keep the same
    pre-build conversion contract here; changing only the resolved config
    would leave a float model while rollout export still quantizes weights.
    """
    quantization = getattr(getattr(config, "extension", None), "quantization", None)
    if quantization is None or not getattr(quantization, "enable_quantized_training", False):
        return config.model_spec

    from .torchao_converter import apply_quantization_converter

    compile_config = getattr(config, "compile", None)
    model_compile_enabled = bool(
        getattr(compile_config, "enable", False)
        and "model" in getattr(compile_config, "components", ())
    )
    logger.info(
        "Applying TorchAO-NPU quantization recipe=%s before VIME model construction",
        quantization.recipe,
    )
    converted = apply_quantization_converter(
        config.model_spec,
        quantization,
        model_compile_enabled=model_compile_enabled,
    )
    config.model_spec = converted
    return converted


def validate_quantized_model(model, config) -> None:
    """Fail fast if the resolved quantized recipe did not reach model modules."""
    quantization = getattr(getattr(config, "extension", None), "quantization", None)
    if quantization is None or not getattr(quantization, "enable_quantized_training", False):
        return
    if quantization.recipe == "all_block_fp8":
        from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor

        expected_type = BlockMXTrainingWeightWrapperTensor
    else:
        from torchao_npu.wrapper_tensors.mx_wrapper_tensor import MXTrainingWeightWrapperTensor

        expected_type = MXTrainingWeightWrapperTensor
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("attention.wq_a")
    ]
    if not targets:
        raise RuntimeError("Quantized training was enabled, but no DSV4 attention.wq_a module was built")
    unconverted = []
    for name, module in targets:
        weight = module.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if not isinstance(weight, expected_type):
            unconverted.append(f"{name}.weight={type(weight).__name__}")
    if unconverted:
        raise RuntimeError(
            "Quantized training was enabled, but DSV4 wq_a modules remained unquantized: "
            + ", ".join(unconverted[:4])
        )
    logger.info("Verified %d wq_a weights use %s", len(targets), expected_type.__name__)
