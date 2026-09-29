# Feature Extraction Configuration

This directory contains configuration files for DeepSeek-V4 feature extraction.

## Files

- **extract_deepseek_v4.yaml** - A3 W8A8-INT8 configuration (8 NPUs by default)

## Configuration Template

```yaml
model_config:
  model_name: "deepseek_v4"
  model_path: "/data/models/deepseek_v4_int8_w8a8"  # A3 W8A8-INT8 export
  platform_version: "A3"                             # Atlas A3 / 910_93
  exe_mode: "eager"                       # CRITICAL: Must be eager!
  enable_cache_compile: False

parallel_config:
  world_size: 8          # Number of NPUs (set to 16 for a 16-rank A3 job)
  moe_ep_size: 8         # MoE expert parallelism (match deployment)

scheduler_config:
  batch_size: 32         # Adjust based on memory

feature_extraction:
  target_layer_ids: [40, 41, 42]  # Layers to capture
  min_loss_tokens: 14              # Min assistant tokens
  max_length: 4096                 # Max sequence length
```

## Platform-Specific Settings

### 950 Platform (8 NPUs)

The default online producer targets Atlas A3. For 950 deployments, use a
950-compatible export and set `platform_version: "950"` explicitly.

```yaml
model_config:
  model_path: "/path/to/950-compatible-model"
  platform_version: "950"

parallel_config:
  world_size: 8
  moe_ep_size: 8
```

### A3 Platform (16 NPUs)
```yaml
model_config:
  # Must be produced with models/deepseek_v4/utils/convert_model.py
  # --quant_type w8a8-int; do not point this at the original FP8 checkpoint.
  model_path: "/data/models/deepseek_v4_int8_w8a8"
  platform_version: "A3"

parallel_config:
  world_size: 16
  moe_ep_size: 16
```

## Critical Settings

**exe_mode: "eager"** - MUST be set to "eager". Forward hooks do NOT work in npugraph_ex mode.

**enable_cache_compile: False** - Disable cache compilation for hooks to work properly.

## Adjusting for Memory

If encountering OOM errors:

1. Reduce `batch_size`: 32 → 16 → 8
2. Reduce `max_length`: 4096 → 2048 → 1024
3. Capture fewer layers: 3 → 2 → 1
4. Reduce `moe_ep_size` if applicable
