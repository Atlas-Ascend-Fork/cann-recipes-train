# torchtitan 启动器

Ascend NPU 上以 torchtitan 后端 + vLLM rollout 跑 RL 训练的启动脚本。

```
scripts/torchtitan/
  run-qwen3-4B.sh              入口，训练参数
  common/launcher.sh       机制：解析、校验、组装、提交
  common/default.sh        开关与环境的默认值
  common/local.sh          本机路径与实验覆盖（不上库）
  common/local.sh.example  模板
```

## 启动

```bash
bash scripts/torchtitan/run-qwen3-4B.sh
```

首次使用需先建本机配置：复制 `common/local.sh.example` 为 `common/local.sh`，填入路径、CANN 位置、venv 与可见卡。不建也能跑，但路径多半不对。要多份配置时按 `local.<experiment>.sh` 各存一份，用 `--local` 切换。

命令行只有三个参数，其余设定都在文件里：

```
--local <name>        用 common/ 下的哪份配置，一台机器或一组实验参数一份，默认 local.sh
--dump-dir <path>     落 rollout 数据
--replay-dir <path>   回放已落的数据（与 --dump-dir 互斥）
```

`--` 之后的内容原样追加到训练命令行末尾，因此覆盖同名参数：

```bash
bash scripts/torchtitan/run-qwen3-4B.sh --local local.<experiment>.sh -- --num-rollout 5 --training.max_norm 0.5
```

---

## 1. 参数分类与优先级

为了方便本地根据不同环境和实验目的进行参数修改，需要将参数进行分类，在不同切换环境或者模型模型时可以更方便快捷。

| 类别 | 内容 | 位置 | 何时改动 |
|---|---|---|---|
| **训练参数** | `train.py` 入参 | `run-qwen3-4B.sh` | 修改默认参数 |
| **开关** | 控制哪些训练参数被加入 | `common/default.sh` | 新增参数 |
| **环境** | 路径、卡、端口、ray worker 环境变量 | `common/default.sh` | 新增参数 |

local.sh 中可以对上述配置进行临时覆盖，可以有多个 local.sh 脚本，运行时按 --local 指定进行切换，`common/local.sh` 在 `common/default.sh` **之前** source，后者只填补尚未设置的项并派生其余。因此在 `local.sh` 里设 `VIME_WORKSPACE`，由它派生的七个路径会一起改变。开关与环境在 `common/default.sh` 里一律写作 `${VAR:-默认值}`，已设过的值优先；训练参数通过追加覆盖。
优先级：

```
命令行  >  环境变量 / local.sh  >  common/default.sh / run-qwen3-4B.sh
```

训练参数没有环境变量这一档（bash 数组无法 `export`），其余相同。


---

## 2. 脚本说明

### 2.1 run-qwen3-4B.sh

入口脚本，配置训练参数。其中参数设置放在两个 launcher 调用之间：

```bash
source .../common/launcher.sh
configure_run "$@"     # local.sh、default.sh 参数加载等

  ... 参数声明 ...

start_run              # 校验、组装、启动
```

参数按用途分成具名数组，最后拼进 `TRAIN_ARGS`，拼接顺序即命令行顺序：

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip      0.2
   --eps-clip-high 0.28
)
add_when TIS_ON 1 GRPO_ARGS -- --use-tis --tis-clip 2.0 --tis-clip-low 0.0

TRAIN_ARGS=(
   "${PLACEMENT_ARGS[@]}" "${ASYNC_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}"
   "${VLLM_ARGS[@]}"      "${EVAL_ARGS[@]}"  "${PERF_ARGS[@]}" "${GRPO_ARGS[@]}"
   "${RECIPE_ARGS[@]}"    "${TB_ARGS[@]}"    "${MISC_ARGS[@]}"
)
```

`add_when <变量> <取值|nonempty> <数组> -- <参数...>` 在开关匹配时追加，`gate_when <变量> <取值> <数组>` 在不匹配时清空整组。

加新参数时，用到不止一处、或 launcher 要读它做校验的，才提取成变量，其余就地写在使用处。现有的提取项：`SEQ_LEN` 供 rollout 上下文与 vLLM 的 max-model-len 使用（torchtitan 的 `training.seq_len` 由后端从 `--rollout-max-context-len` 推出，脚本里不写）；`GLOBAL_BATCH` 是另外两个的乘积；`MAX_RESPONSE_LEN` 与 `VLLM_RANKS_PER_ENGINE` 参与校验。

`RECIPE_ARGS` 里的 `--module` / `--config` 指向 torchtitan 预置配置函数（`vime/backends/torchtitan_utils/models/<族>/config_registry.py`）。tyro 表达不了的配置放在预置配置里：模型 flavor 与 attention 后端、优化器超参（上游把 lr/betas/weight_decay 挪进了 `ParamGroupConfig.optimizer_kwargs`）、精度三件套、激活重计算策略。要改这些就改预置配置函数，其余仍在本脚本用 `--section.key` 覆盖。

`⚠` 标记改动后会出问题的值：精度设置、与 megatron 对齐的取值。

训练与 rollout 的卡分配写在 `PLACEMENT_ARGS` 里。当前是两者各占一半、互不相交；改成共卡则把两者都设为可见卡数并加 `--colocate`。

开新实验的方式是复制这个文件：`RUN_NAME` 默认取自文件名，决定 ckpt、TensorBoard 与归档目录的名字。

### 2.2 default.sh

开关与环境的默认值，分两段。它同时就是 `local.sh` 可配置项的清单——每一行都是 `${VAR:-默认值}`，在 `local.sh` 里 `export` 同名变量即可覆盖任意一项。

三个根决定其余：

| 根 | 派生出 |
|---|---|
| `VIME_WORKSPACE` | `VIME_DATA_ROOT`、`VIME_CKPT_ROOT`、`VIME_TB_ROOT`、`TMPDIR`、`TRITON_CACHE_DIR`、`HF_HOME`、`ASCEND_CACHE_PATH` |
| `VIME_SLOT` | 五个端口与 `VIME_RAY_TEMP_DIR` |
| `VIME_TRAIN_ENV` / `VIME_VLLM_ENV` | `TRAIN_PYTHON`、`VIME_VLLM_EXECUTABLE`、`VIME_VLLM_PYTHON` |

文件末尾的 `RT_ENV` 是交给 ray worker 的环境变量：NPU 与 vLLM 的各项规避设置，每项都注明了存在的原因。launcher 另外追加计算得出的四项（`TENSORBOARD_DIR`、`PYTHONPATH`、`VIME_VLLM_EXECUTABLE`、`VIME_VLLM_PYTHON`）与确定性相关项。

### 2.3 local.sh

本机取值，不上库。复制 `local.sh.example` 后填写。可保留多份，运行时用 `--local` 指定；不带路径的名字相对 `common/` 解析，指定了不存在的文件会报错退出。

可设置的全部参数：

| 变量 | 默认值 | 说明 |
|---|---|---|
| **根** | | |
| `VIME_WORKSPACE` | `/tmp/vime` | data / ckpt / tb / tmp / 各类缓存的共同根 |
| `VIME_MODEL_ROOT` | `/tmp/models` | 模型库，通常与 workspace 不同盘 |
| `VIME_SLOT` | `0` | 同机第几个并发实例，决定整套端口 |
| **运行时** | | |
| `VIME_CANN_HOME` | `/usr/local/Ascend/ascend-toolkit` | 需含 `set_env.sh` |
| `VIME_NNAL_HOME` | `/usr/local/Ascend/nnal/atb` | 需含 `set_env.sh` |
| `VIME_TRAIN_ENV` | 空 | 训练侧与 vime 侧 python 环境；留空则用当前解释器 |
| `VIME_VLLM_ENV` | 同 `VIME_TRAIN_ENV` | vLLM 侧 python 环境；设成不同环境即双环境 |
| `TRAIN_PYTHON` | 由 `VIME_TRAIN_ENV` 推导 | 直接点名可执行文件，绕过推导 |
| `VIME_VLLM_EXECUTABLE` | 由 `VIME_VLLM_ENV` 推导 | 同上；会传给 ray worker，serve 子进程据此切换解释器 |
| `VIME_VLLM_PYTHON` | 由 `VIME_VLLM_ENV` 推导 | vLLM serve 子进程用于 re-exec 的 Python；双环境时不要指向 shell wrapper |
| `VIME_NOFILE_LIMIT` | `65535` | `ulimit -n`；过低会让 raylet 因 EMFILE 中止 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,…,7` | 本机可用的卡；如何分给训练与 rollout 在 `run-qwen3-4B.sh` |
| **开关** | | |
| `MODE` | `sync` | `sync` \| `async` \| `fully-async` |
| `DET` | `on` | 一次开关三处确定性 |
| `EVAL_ON` | 跟随 `MODE` | 设 `off` 可在测吞吐或冒烟时跳过 eval |
| **派生路径** | | 布局不同时可单独覆盖 |
| `VIME_DATA_ROOT` | `$VIME_WORKSPACE/data` | |
| `VIME_CKPT_ROOT` | `$VIME_WORKSPACE/ckpt` | |
| `VIME_TB_ROOT` | `$VIME_WORKSPACE/tb` | |
| `TMPDIR` | `$VIME_WORKSPACE/tmp` | |
| `TRITON_CACHE_DIR` | `$VIME_WORKSPACE/.triton` | |
| `HF_HOME` | `$VIME_WORKSPACE/.cache/huggingface` | |
| `ASCEND_CACHE_PATH` | `$VIME_WORKSPACE/.ascend_cache` | |
| **派生端口** | | 需要对上既有配置时可单独覆盖 |
| `VIME_RAY_PORT` | `6379 + VIME_SLOT` | |
| `VIME_RAY_DASHBOARD_PORT` | `8265 + VIME_SLOT` | |
| `VIME_RAY_DASHBOARD_AGENT_PORT` | `52365 + VIME_SLOT` | ray 的默认值在同机多集群时必撞 |
| `HCCL_HOST_SOCKET_PORT_RANGE` | `60000+100·SLOT` 起 50 个 | |
| `HCCL_NPU_SOCKET_PORT_RANGE` | `61000+100·SLOT` 起 50 个 | |
| `VIME_RAY_TEMP_DIR` | `$TMPDIR/ray-s$VIME_SLOT` | 刻意取短：AF_UNIX 路径上限 107 字节 |
| `MASTER_ADDR` | `127.0.0.1` | |
| **训练参数** | | |
| `OVERRIDE_ARGS` | 空 | 数组，追加在所有参数组之后，因而覆盖同名项 |

dual-venv，训练与 rollout 各用一套 torch：

```bash
export VIME_TRAIN_ENV=/path/to/train/env
export VIME_VLLM_ENV=/path/to/vllm/env
```

按机器覆盖某个训练参数：

```bash
OVERRIDE_ARGS=(--num-rollout 3000 --max-tokens-per-gpu 4096)
```

---

## 3. launcher 派生的内容

### 模式

`MODE` 展开成五处改动：

| | `sync` | `async` | `fully-async` |
|---|---|---|---|
| 入口脚本 | `train.py` | `train_async.py` | `train_async.py` |
| `--update-weights-interval 1` | — | 有 | 有 |
| TIS 参数 | — | 有 | 有 |
| `--rollout-function-path` | — | — | 全异步 rollout |
| eval 参数 | 有 | 有 | 丢弃 |

`fully-async` 丢弃 eval，因为 `generate_rollout_fully_async` 在 eval 模式直接 raise。此时若显式设 `EVAL_ON=on`，启动前即报错。

### 确定性

`DET=on` 在三处同时加入参数：训练侧的 `--debug.seed` 与 `--debug.deterministic`、采样侧的 `--vllm-enable-deterministic-inference` 与 `--rollout-seed`、以及 ray worker 环境里的 `HCCL_DETERMINISTIC=strict` 与 `LCCL_DETERMINISTIC=1`。

之所以是一个开关而非三个：HCCL 这项必须两端一致。vllm-ascend 的 batch-invariant 路径会把 rollout worker 强制设为 `strict`，训练侧若停在 `true`，跨进程权重传输所用的集合通信会死锁。

### 目录

```
ckpt   $VIME_CKPT_ROOT/<run name>-<mode>
归档   $VIME_TB_ROOT/<run name>/<时间戳>
```

ckpt 路径跨次运行稳定，`--load` 才能续上；归档路径带时间戳，多次运行的 TensorBoard 事件不会互相覆盖。`--tb-experiment-name` 取 `<run name>-<时间戳>`。

每次运行在归档目录下写两个文件：

| 文件 | 内容 |
|---|---|
| `rerun.sh` | 可执行：`ray start` 与 `ray job submit`，参数与本次运行完全相同 |
| `train.log` | 训练输出 |

`rerun.sh` 存的是字面命令而非重新调用模型脚本，因此之后修改参数或 `local.sh` 都不会改变它重放的内容。

### ray worker 环境变量

在 `RT_ENV` 中声明一次，两处消费：export 进本进程，以及序列化进 `--runtime-env-json`。JSON 由 `json.dumps` 生成，如果用字符串拼接的方式，`LD_PRELOAD` 与 `PYTHONPATH` 可能含需转义的字符。

### 校验

提交前检查：

| 检查项 | 原因 |
|---|---|
| `--dump-dir` 与 `--replay-dir` 互斥 | 一个写一个读 |
| `MODE=fully-async` 时不得 `EVAL_ON=on` | 该 rollout 在 eval 模式 raise |
| `MAX_RESPONSE_LEN < SEQ_LEN` | 后者是 prompt 与 response 的总上界 |
| `VLLM_RANKS_PER_ENGINE` 整除 `ROLLOUT_GPUS` | vime 按整除计算引擎数，除不尽的卡会闲置 |
| `VIME_RAY_TEMP_DIR` 长度 | ray 的 AF_UNIX socket 路径上限 107 字节 |
| CANN、nnal、venv 路径存在 | 否则故障现场离真实原因很远 |
