# Vime

[中文版](./README_zh.md) · [Repository](https://github.com/vllm-project/vime)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.vllm.ai/projects/vime/en/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vllm-project/vime)

**Vime** is an LLM post-training framework for RL scaling, built on [slime](https://github.com/THUDM/slime). It keeps slime's training stack and data-generation design while using [**vLLM**](https://github.com/vllm-project/vllm) (with [vllm-router](https://github.com/vllm-project/router)) as the default rollout backend. Built on top of Vime, this repository targets Ascend NPU and uses [**torchtitan**](https://github.com/pytorch/torchtitan) as the training backend, providing two core capabilities:

1. **High-performance training**: Efficient training in various modes by connecting torchtitan with vLLM;
2. **Flexible data generation**: Arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

> **Maintenance note**: torchtitan is the primary training backend of this repository. The megatron backend currently supports a subset of the Qwen3 series and will no longer be maintained.

## Positioning

The vLLM community horizontally supports many LLM post-training frameworks, including (in alphabetical order) [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [OpenRLHF](https://github.com/openrlhf/openrlhf), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [SkyRL](https://github.com/NovaSky-AI/SkyRL), [verl](https://github.com/verl-project/verl), and so on. We built the Vime project to seamlessly bring slime's proven training paradigm into the vLLM ecosystem, offering a production-ready bridge that aligns both projects' rapid release cycles. We hope that users with different needs can find the right vLLM-ecosystem choice for their workflows. The vLLM community will continue to support the vLLM integration in these post-training frameworks. This repository is the Ascend NPU version: the rollout side reuses Vime's vLLM integration, while the training side replaces megatron with torchtitan — model definitions, FSDP2 parallelism, and DCP checkpoints all come from upstream torchtitan, and the training workflow and interfaces stay consistent with Vime.

## Table of Contents

- [Vime](#vime)
  - [Positioning](#positioning)
  - [Table of Contents](#table-of-contents)
  - [Supported Models and Hardware](#supported-models-and-hardware)
  - [Architecture Overview](#architecture-overview)
  - [Quick Start](#quick-start)
    - [Single-turn AscendC operator generation](#single-turn-ascendc-operator-generation)
    - [Agentic RL examples](#agentic-rl-examples)
  - [Arguments Walkthrough](#arguments-walkthrough)
  - [Engine Deployment](#engine-deployment)
  - [Correctness, Stability, and CI](#correctness-stability-and-ci)
  - [Code Reading Path](#code-reading-path)
  - [Developer Guide](#developer-guide)
  - [slime doc](#slime-doc)
  - [FAQ](#faq)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Supported Models and Hardware

| Model | torchtitan preset config function | Hardware |
|---|---|---|
| Qwen3-4B | `vime_qwen3_4b` | Atlas 800T A3, A5 |
| Qwen3-30B-A3B | `vime_qwen3_30b_a3b` | Atlas 800T A3, A5 |
| DeepSeek V4 Flash | `vime_dsv4_flash` | A5 |

Preset configs live in `vime/backends/torchtitan_utils/models/<model family>/config_registry.py`.

## Architecture Overview

![arch](./imgs/arch_torchtitan.svg)

**Module Descriptions**:

- **training (torchtitan)**: Runs the main training loop, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
- **rollout (vLLM + router)**: Launches vLLM inference engines and routes generation requests; custom generate functions can wrap them with multi-turn loops, tool calls, environment/sandbox interaction, and verifier-based rewards.
- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods, including agentic workflows that produce samples through the same interface.

## Quick Start

For training startup, config overrides, and log troubleshooting, see the [torchtitan launcher](scripts/torchtitan/README.md). For the training backend's run modes, configuration, and data processing flow, see the [torchtitan training backend](docs/torchtitan-backend.md).

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

### Single-turn AscendC operator generation

[`examples/cann_bench_single_turn`](examples/cann_bench_single_turn/README.md) provides TorchTitan NPU launch configs for Qwen3.5-4B/9B and Qwen3.6-27B, using native single-turn generation and an online reward to plug into CANN-Bench v1 evaluation. See the example docs for data preparation, run steps, and each model's validation boundary; it does not depend on the agent sandbox. The standalone [evaluation client](examples/cann_bench_evaluator/README.md) can be reused by other tasks.

### Agentic RL examples

Agentic workloads use the standard rollout / Data Buffer loop through Vime's customization interfaces; they are not a separate framework:

- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md): End-to-end coding-agent RL with Claude Code or Codex, sandboxed tool use, test-based rewards, and token-correct trajectory segments.

See the [Agentic RL Training Roadmap](docs/en/get_started/agent.md) and [Customization Guide](docs/en/get_started/customization.md). The coding-agent example ships an E2B-compatible backend, while the shared `vime.agent.sandbox.Sandbox` contract can be implemented for Docker, Modal, or local VMs.

## Arguments Walkthrough

Arguments in Vime are divided into three categories:

1. **torchtitan arguments**: `--module` / `--config` selects a preset config function from `config_registry`, which returns a preset config providing baselines such as model specification, optimizer, and precision; dotted arguments of the form `--<field>.<subfield>` override fields in the preset config, for example `--parallelism.tensor_parallel_degree 4`. See Chapter 6 of the [torchtitan training backend](docs/torchtitan-backend.md).
2. **vLLM arguments**: vLLM server and engine options are exposed with a `--vllm-` prefix (for example, `--vllm-gpu-memory-utilization`). vllm-router's own options are passed with `--router-` (for example, `--router-policy round_robin`); Vime-side router orchestration arguments use the `--vllm-router-` prefix, including `--vllm-router-ip`, `--vllm-router-port`, and the effective `--vllm-router-request-timeout-secs`. See [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py) for the full surface.
3. **Framework arguments**: Flags related to Vime orchestration (rollout GPUs, data paths, RL algorithms, etc.), see [vime/utils/arguments.py](vime/utils/arguments.py).

`--rollout-num-gpus-per-engine` sets the tensor parallel size of each vLLM engine. The default rollout entry is `vime.rollout.vllm_rollout.generate_rollout`.

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Engine Deployment

Vime organizes the RL dataflow on top of the native torchtitan and vLLM control surfaces. Beyond the argument pass-through described above, see:

- [vLLM Config](docs/en/advanced/vllm-config.md): Optional YAML topology configuration, heterogeneous server groups, multi-model serving, and per-group overrides;
- [PD Disaggregation](docs/en/advanced/pd-disaggregation.md): For multi-turn and agentic workloads with different prefill/decode resource needs;
- Router policies such as session affinity for multi-turn agents (see [vLLM Config](docs/en/advanced/vllm-config.md));
- [Delta Weight Sync](docs/en/advanced/delta-weight-sync.md): Disk-based incremental updates for disaggregated rollout engines;
- [External Rollout Engines](docs/en/advanced/external-rollout-engines.md): Serving managed outside the training job. Serving can use an independent environment; disk transport avoids an NCCL group between training and serving. Different GPU models or vendors still require compatible model formats, precision, and vLLM hardware support.

## Correctness, Stability, and CI

RL bugs can be silent. Vime keeps the dataflow explicit and supports separate rollout-only and train-only debugging paths. CPU unit tests, customization-hook contract tests, and GPU end-to-end suites protect different parts of this workflow. Buildkite runs always-on CPU checks; GPU suites require the manual gate, so a green CPU build is not GPU validation.

Useful engineering docs:

- [CI](docs/en/developer_guide/ci.md)
- [Debugging](docs/en/developer_guide/debug.md)
- [Reproducibility](docs/en/advanced/reproducibility.md)
- [Fault Tolerance](docs/en/advanced/fault-tolerance.md)
- [Trace Viewer](docs/en/developer_guide/trace.md)
- [Profiling](docs/en/developer_guide/profiling.md)

## Code Reading Path

Start from the training loop and follow the calls only as deep as needed:

```text
train.py: train
├─ vime/ray/placement_group.py       Ray resource and worker initialization
├─ vime/ray/rollout.py              RolloutManager.generate: rollout orchestration
│  └─ vime/rollout/vllm_rollout.py  Sample generation and reward computation
└─ vime/ray/actor_group.py          RayTrainGroup.async_train: training dispatch
   └─ vime/backends/torchtitan_utils/actor.py
      ├─ engine.py                   torchtitan model execution
      └─ loss.py                     RL losses and advantages
```

On a first pass, treat `vime/utils/arguments.py` as the configuration entry point. The deployment details in `vime/backends/vllm_utils/` and the weight-sync implementations under `vime/backends/torchtitan_utils/update_weight/` can wait until you need to change those areas.

## Developer Guide

- **Contributions are welcome!** If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR.

- Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

  ```bash
  apt install pre-commit -y
  pre-commit install

  # run pre-commit to ensure code style consistency
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```

- For debugging tips, please refer to the [Debugging Guide](docs/en/developer_guide/debug.md)

## slime doc

Vime is derived from slime. The following upstream resources and in-repo guides still use the slime naming and remain the reference for shared concepts (torchtitan integration, customization, advanced topics):

[![Documentation](https://img.shields.io/badge/slime_docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

- Upstream repository: [THUDM/slime](https://github.com/THUDM/slime)
- English docs in this repo: [docs/en/](docs/en/)
- Chinese docs in this repo: [docs/zh/](docs/zh/)

## FAQ

For frequently asked questions, please see the [Q&A](docs/en/get_started/qa.md)

## Acknowledgements

Vime builds on ideas and infrastructure from the open-source RL ecosystem. We especially thank the [slime](https://github.com/THUDM/slime) community, whose great work Vime is directly built on. We also thank [SkyRL](https://github.com/NovaSky-AI/SkyRL) and [verl](https://github.com/verl-project/verl), whose excellent work we referenced. Vime is maintained by the vLLM community.

## Citation

```bibtex
@misc{vime,
  author       = {Vime Contributors},
  title        = {Vime: An LLM post-training framework with vLLM for RL Scaling},
  year         = {2026},
  howpublished = {\url{https://github.com/vllm-project/vime}},
  urldate      = {2026-06}
}
```
