# 基于 vLLM Speculators 的 GLM 5.2 DSpark 草稿模型 NPU 训练样例

## 1. 概述

本样例基于 [vllm-project/speculators](https://github.com/vllm-project/speculators) 训练框架，在昇腾 NPU 上在线训练 GLM 5.2-w4a8c8 的 **DSpark 草稿模型**（draft model）。草稿模型用于投机解码（speculative decoding）推理加速：推理时先由轻量草稿模型一次生成一个候选块（draft block），再由 verifier 模型单次前向并行验证，在不改变输出分布的前提下提升生成吞吐。DSpark 在 DFlash 基础上增加了建模块内 token 依赖的 Markov Head 与逐位置接受概率预测的 Confidence Head。

**评测结果**（投机解码平均接受长度 AL，每个 verifier 前向接受的平均 token 数，越高加速比越高）：

| 类别 | 数据集 | AL |
| --- | --- | --- |
| 数学 | GSM8K | 6.3 |
| 数学 | MATH500 | 6.33 |
| 代码 | HumanEval | 4.91 |
| 代码 | MBPP | 4.58 |
| 通用中文 | Magpie | 3.65 |
| 通用 | Alpaca | 3.87 |
| 通用 | MTBench | 4.1 |
| 长序列 | Longbench-v2 | 4.48 |
| Agent | 自研Agent数据集\* | 3.82 |

\* 自研 Agent 评测数据集（非公开）。

hidden-states 传输走 **mooncake** backend：引擎把 hidden-states 直接写进 mooncake 分布式 store，训练器按同一 master 地址从 store 取，用完即删。

训练得到的草稿模型 checkpoint 可用于 vLLM-Ascend 投机解码推理，benchmark 验证与部署对接 `cann-recipes-infer` 仓。

## 2. 使用的产品型号

| 项目 | 规格 |
| --- | --- |
| HS 提取侧 | Atlas A3，DP8×TP4 大 EP |
| 训练侧 | Atlas A2（910B3），FSDP |
| 操作系统 | Ubuntu 22.04.5 LTS（ARM64） |
| Python | 3.12 |
| torch / torch_npu | 2.10.0（CPU wheel） / 2.10.0.post2 |
| CANN | 9.1.0 |
| vLLM | 0.26.0 |
| vllm-ascend | 0.26.0rc1 |

## 3. 文件说明

|文件/目录|说明|
|---|---|
|`glm5.2/README.md`|训练说明（本文件）|
|`scripts/train_config_glm5.2.yaml`|训练配置：模型路径、ENGINES 端点表、双 master 映射与专用 flag（平铺 yaml，启动脚本按文件名自动读取，同名环境变量优先）|
|`scripts/train_dspark_glm5.2_online.sh`|在线训练启动脚本（后台运行；按文件名自动 source 上述 config，torchrun 目标为上述 wrapper）|
|`scripts/serve_vllm_hs_glm5.2.sh`|HS 引擎节点一键启动：拉起本机 4 个 `vllm serve`（mooncake connector）；两台用同一份文件，按本机 IP 与 dp-rank 传参|
|`scripts/vllm_regenerate_glm5.2.sh`|启动 §6.3 中文数据重生成用的 vLLM chat 服务（无 hidden-states 提取）|
|`scripts/route_rank_endpoints_glm5_2.py`|torchrun 目标：按 `LOCAL_RANK` 逐 rank 注入 `--vllm-endpoint`（本样例专用；hidden-states 走 mooncake，不需要注入任何 hs 地址）|
|`utils/opb_regen_to_tokenized_jsonl.py`|open-perfectblend regen 的 parquet → 对话 jsonl + 已分词 jsonl（本地分词并打监督掩码，不需要引擎）|
|`../utils/select_cmb_chinese_prompts.py`|CMB-Eagle-Mix-330k → 中文 prompt 集合（speculators/utils/ 共用脚本，按中文占比降序取前 N 条，§6.3）|
|`utils/derotate_draft_weights.py`|发布前对草稿 `fc` 反向旋转（de-QuaRot，§9）|
|`requirements.txt`|训练侧依赖（锁定版本）|

## 4. 源码准备

本节假设 `cann-recipes-train` 与 `speculators` 克隆在同一级目录；不是这种摆法的话，把那些路径换成自己的实际位置。

```bash
# 1. 克隆原仓并固定到验证过的基线 commit d3c3306（已含上游 PR #1224：fc 输入
#    布局，DSpark 多层取层靠它，无需任何环境变量开关；本样例取层 5 23 42 60 78
#    含末层（id 78），必须含该 PR；不含末层的取层则无需它）
git clone https://github.com/vllm-project/speculators.git
cd speculators
git checkout d3c3306

# 2. 先可编辑安装子包 hs_connectors，再安装本体
cd hs_connectors && pip install -e . && cd ..
pip install -e .

# 3. 将脚本拷入仓库 scripts/ 与 utils/（SAMPLE = 本样例目录）
SAMPLE=../cann-recipes-train/llm_spec/speculators/glm5.2
cp $SAMPLE/scripts/route_rank_endpoints_glm5_2.py \
   $SAMPLE/scripts/train_dspark_glm5.2_online.sh \
   $SAMPLE/scripts/train_config_glm5.2.yaml \
   $SAMPLE/scripts/serve_vllm_hs_glm5.2.sh \
   $SAMPLE/scripts/vllm_regenerate_glm5.2.sh scripts/
mkdir -p utils
cp $SAMPLE/utils/opb_regen_to_tokenized_jsonl.py \
   $SAMPLE/utils/derotate_draft_weights.py utils/
cp $SAMPLE/../utils/endpoint_routes.py utils/
```

## 5. 环境准备

### 5.1 CANN 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

未 source 时 vllm-ascend 启动会报 `Engine core initialization failed: No module named 'acl'`。`serve_vllm_hs_glm5.2.sh` 在拉起引擎前会显式 source（幂等，可重复执行）。

### 5.2 训练侧依赖

```bash
pip install -r ../cann-recipes-train/llm_spec/speculators/glm5.2/requirements.txt
```

hidden-states 走 mooncake 传输，**训练机与两台引擎节点都要装同一个版本**的 mooncake 传输引擎：

```bash
pip install mooncake-transfer-engine-npu==0.3.11.post1
```

### 5.3 版本要求

| 组件 | 版本要求 | 说明 |
| --- | --- | --- |
| vllm-ascend | ≥ 0.26.0rc1 | hidden-state 层 / hybrid 组 → 私有连续层 buffer，0 NaN |
| vLLM | ≥ v0.26.0 | v0.31.0 另收 abort 安全修复 [vllm#56841](https://github.com/vllm-project/vllm/pull/56841) |
| CANN | ≥ 9.1.0 | — |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 起 | CPU wheel，NPU 由 torch_npu 提供 |
| transformers | 5.14.1 | — |
| mooncake-transfer-engine-npu | 0.3.11.post1 | 训练机与两台引擎节点须同版本；自带 `mooncake_master`。启动 master 时**必须**带 `--default_kv_lease_ttl 300000`（默认 5 秒租约撑不住 8–500 MB 的单笔 hidden-states，见 §7.1） |

W4A8C8 verifier 另需引擎侧支持 `--quantization ascend`，并保证 `/data/GLM-5.2-w4a8c8` 这类量化权重目录在**每台引擎节点本地都可读**。

**A3 侧开关与版本的关系**：`serve_vllm_hs_glm5.2.sh` 里有一组只服务于 A3（910C）大 EP 布局的开关，它们不是通用必需项，版本前提各不相同：

| 开关 | 读取方 | 版本前提 |
| --- | --- | --- |
| `ASCEND_A3_ENABLE=1` | CANN / ACL 侧（A3 机型专用内核与算子路径） | 依赖 CANN 对 A3 的支持，本样例下限即 CANN ≥ 9.1.0 |
| `HCCL_OP_EXPANSION_MODE=AIV` | HCCL（把集合通信算子展开到 AI Vector core） | HCCL 随 CANN 一起提供，同上 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | vllm-ascend（fused MC2 通信算子） | vllm-ascend ≥ 0.26.0rc |
| `--additional-config` 的 `enable_dsa_cp` / `enable_sparse_li_c8` / `enable_sparse_sfa_c8` / `multistream_overlap_shared_expert` / `c8_enable_reshape_optim` / `recompute_scheduler_enable` | vllm-ascend（attention / MoE 路径选择） | `enable_sparse_li_c8` / `enable_sparse_sfa_c8` 自 vllm-ascend v0.26.0rc 起才有，更早版本不认这两个字段 |

本样例只在 §5.3 这张表的版本组合（CANN ≥ 9.1.0、vllm-ascend ≥ 0.26.0rc1、vLLM ≥ v0.26.0）上验证过；换到非 A3 机型时，上表中的 A3 开关与 `additional-config` 字段应一并去掉。

另注：`VLLM_ASCEND_ENABLE_FUSED_MC2=1` 与 `multistream_overlap_shared_expert: true` 不能同时生效，vllm-ascend 会把后者强制关掉并打一条 warning（fused MC2 优先），本样例按后者关闭运行。

## 6. 数据准备

把 open-perfectblend 的 GLM-5.2 regen 版本 prepare 成 arrow（`--seq-length` 须与 §8 的 `--total-seq-len` 一致）。全部在已 `pip install -e .` 的 speculators 仓库根目录执行。

### 6.1 数据下载

```bash
# 公开的 GLM-5.2 regen 版本，ShareGPT 风格 parquet，1,420,229 条，直接拿来用，不需要再重生成
git clone https://huggingface.co/datasets/mgoin/open-perfectblend-glm5.2-regen
```

### 6.2 open-perfectblend → arrow

由于 `prepare-data --data` 只能输入 `.json/.jsonl`，所以先转格式，同时用目标模型自己的 chat template 做本地分词、按 assistant 轮边界打监督掩码：

```bash
python utils/opb_regen_to_tokenized_jsonl.py \
    --src ./open-perfectblend-glm5.2-regen \
    --tokenizer /path/to/GLM-5.2-W4A8C8 \
    --out-conversations ./data/openperfectblend_regen.jsonl \
    --out-tokenized ./data/openperfectblend_regen_tokenized.jsonl \
    --max-samples 600000 --seed 42 --seq-length 8192
```

`--max-samples` 按 `--seed` 等概率抽样，不是取前 N 行；`--out-conversations` 是中间产物，保留着便于核对；`--seq-length` 要与下面 prepare 的取值一致。

```bash
# 行里已带 input_ids/loss_mask，prepare 直接沿用：不渲染、不需要服务
python -m speculators prepare-data \
    --model /path/to/GLM-5.2-W4A8C8 \
    --data ./data/openperfectblend_regen_tokenized.jsonl \
    --output ./output/arrow-opb-600k --seq-length 8192
```

### 6.3 中文数据（CMB 中文子集 10w）

为了提升中文数据的训练效果，本样例在 open-perfectblend 之外还混入了中文数据，取自 [CMBTech/CMB-Eagle-Mix-330k](https://www.modelscope.cn/datasets/CMBTech/CMB-Eagle-Mix-330k)（ModelScope，Apache-2.0）：33w 条中英混合，其中文部分即 [swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT](https://modelscope.cn/datasets/swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT) 的蒸馏数据。选取方式为**按 prompt 的中文占比降序取前 10w**（占比阈值 24%，分母=总字符数，中位 87.2%），每条只取第一个 human 轮作 prompt（原 assistant 轮丢弃，由 target 重新生成，on-policy）；再用 GLM-5.2-W4A8C8 重生成**一遍**并用 GLM-5.2 tokenizer 分词为 input_ids/loss_mask jsonl（regen 的 `--dataset` 直接传本地 .jsonl，原 assistant 轮被丢弃、只留 user prompt 重新生成；服务用 `scripts/vllm_regenerate_glm5.2.sh` 起一个即可，参数沿用 §7 提取引擎、仅去掉提取相关配置——当时实跑打的是远端大 EP 服务，本地脚本为等效替代）。20w 由 prepare 时重复传入同一文件实现（复制加权 2×），与 60w 英文 open-perfectblend 一起构成 80w 训练集。

按 GLM-5.2 tokenizer 计，这批 prompt 长度 p50=23 token、p90=67 token、max=2168 token，超 8192 token 的为 0 条，所以下面的混合 prepare 用 `--seq-length 8192` 不会截断中文 prompt。

```bash
# 0. 启动重生成用 vLLM 服务（另开终端；英文走 §6.1 公开数据集，无需此服务）
MODEL_PATH=/path/to/GLM-5.2-W4A8C8 bash scripts/vllm_regenerate_glm5.2.sh

pip install modelscope
modelscope download --dataset CMBTech/CMB-Eagle-Mix-330k --local_dir ./data/cmb

# 从 33w 条中挑出中文 prompt 前 10w（按中文占比降序，占比阈值 24%，只取首个 human 轮）
python ../utils/select_cmb_chinese_prompts.py \
    --src ./data/cmb/cmb_eagle_mix_330k.jsonl \
    --out ./data/cmb_zh_prompt_100k.jsonl \
    --min-zh-ratio 0.24 \
    --num-prompts 100000

# --endpoint 指向 GLM-5.2 的 chat completions 服务
python -m speculators regenerate-responses \
    --dataset ./data/cmb_zh_prompt_100k.jsonl \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model /path/to/GLM-5.2-W4A8C8 \
    --outfile ./data/zh100k_regen.jsonl \
    --resume \
    --concurrency 64 \
    --max-tokens 8192 \
    --max-retries 3
```

混合 prepare（`--data` 支持多次传入，多文件按序拼接混合）：

```bash
python -m speculators prepare-data \
    --model /path/to/GLM-5.2-W4A8C8 \
    --data ./data/openperfectblend_regen_tokenized.jsonl \
    --data ./data/zh100k_regen.jsonl \
    --data ./data/zh100k_regen.jsonl \
    --output ./output/arrow-opb60w-zh20w --seq-length 8192
```

实际训练消费混合后的 `./output/arrow-opb60w-zh20w`（§8 config 的 `DATA_PATH`）；§6.2 的 `arrow-opb-600k` 是仅英文的中间产物。

## 7. 启动 HS 提取服务（A3 双机 DP8×TP4 大 EP）

分两步：先在训练机上起两个 mooncake master（每台引擎机器一个），再在两台 A3 上各起 4 个引擎（8 个引擎组成 DP8×TP4 大 EP，每个引擎导出 `TARGET_LAYER_IDS` 指定的那几层）。

### 7.1 起 mooncake master（每台引擎机器一个，共两个）

```bash
# master A：给 Node0 那 4 个引擎用
nohup mooncake_master --rpc_port=50051 --metrics_port=9003 --default_kv_lease_ttl 300000 > mooncake_master.50051.log 2>&1 &
# master B：给 Node1 那 4 个引擎用
nohup mooncake_master --rpc_port=50052 --metrics_port=9013 --default_kv_lease_ttl 300000 > mooncake_master.50052.log 2>&1 &
```

### 7.2 起引擎（两台 A3 各跑一次）

```bash
# Node0（DP master）：连 master A
NODE_IP=<Node0 IP> DP_ADDRESS=<Node0 IP> DP_RANK_START=0 \
  MOONCAKE_MASTER=<训练机 IP>:50051 MODEL_PATH=/data/GLM-5.2-w4a8c8 \
  bash scripts/serve_vllm_hs_glm5.2.sh

# Node1：同一条命令，NODE_IP / DP_RANK_START / MOONCAKE_MASTER 不同（连 master B）
NODE_IP=<Node1 IP> DP_ADDRESS=<Node0 IP> DP_RANK_START=4 \
  MOONCAKE_MASTER=<训练机 IP>:50052 MODEL_PATH=/data/GLM-5.2-w4a8c8 \
  bash scripts/serve_vllm_hs_glm5.2.sh
```

可调项（环境变量；`—` 表示必填，没有默认值）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NODE_IP` | — | 本机对外 IP，`HCCL_IF_IP` 与 `MOONCAKE_LOCAL_HOSTNAME` 都用它 |
| `DP_ADDRESS` | — | master 那台的 IP，两台填同一个值 |
| `DP_RANK_START` | — | 本机第一个引擎的 dp-rank：Node0 填 `0`，Node1 填 `DP_SIZE_LOCAL`（4） |
| `MOONCAKE_MASTER` | — | 引擎要连的 mooncake master 地址 |
| `MODEL_PATH` | `/data/GLM-5.2-w4a8c8` | 量化权重目录，两台本地都要有；**这个字符串就是 `GET /v1/models` 的 id，必须与训练侧 `VERIFIER` 一致** |
| `nic_name` | `bond0` | HCCL/GLOO 网卡名 |
| `TARGET_LAYER_IDS` | `5 23 42 60 78` | 取层列表（脚本内转逗号注入 speculative-config），**须与训练侧 `--target-layer-ids` 一致** |
| `MC_HANDSHAKE_PORT_BASE` | `51010` | 本机引擎握手端口起点，每个引擎 +1，**须能被 master 连到** |
| `MOONCAKE_GLOBAL_SEGMENT_GIB` / `MOONCAKE_LOCAL_BUFFER_GIB` | `25` / `4` | 每个引擎注册给 store 的内存（本机 4 引擎合计 100 GiB） |
| `DP_RPC_PORT` | `10521` | 引擎间 DP RPC 端口，两台须填同一个值 |
| `MOONCAKE_WRITER_THREADS` | `8` | 引擎侧异步写线程数 |
| `DP_SIZE` / `DP_SIZE_LOCAL` / `TP_SIZE` | `8` / `4` / `4` | 大 EP 布局 |
| `VLLM_START_PORT` | `6700` | 本机引擎起始端口 |


## 8. 训练

### 8.1 启动

以下变量的默认值已写进 `scripts/train_config_glm5.2.yaml`（平铺 yaml），直接改配置或用环境变量覆盖皆可（启动脚本按自身文件名读取本样例的 config）：

```bash
# 8 个 HS 引擎端点：rank r 用第 r 个（Node0 那 4 个在前，顺序不能换）
# TRAIN_IP 是本机可路由 IP，mooncake 要靠它回连本机的 dataloader worker
# HOST_MASTERS：按"引擎所在机器"给每台一个 master（格式 <引擎机器IP>=<master地址>），
#               与 §7.1 的两个 master 对应，顺序无所谓
ENGINES="<Node0 IP>:6700 <Node0 IP>:6701 <Node0 IP>:6702 <Node0 IP>:6703 \
         <Node1 IP>:6700 <Node1 IP>:6701 <Node1 IP>:6702 <Node1 IP>:6703" \
TRAIN_IP=<训练机 IP> \
HOST_MASTERS="<Node0 IP>=127.0.0.1:50051 <Node1 IP>=127.0.0.1:50052" \
  bash scripts/train_dspark_glm5.2_online.sh
```

torchrun 的目标是 `scripts/route_rank_endpoints_glm5_2.py`，不是直接跑 `speculators.train`：框架的 `--vllm-endpoint` 只接受单值，多端点扇出放在这个 wrapper 里——它按 `LOCAL_RANK` 给每个 rank 注入自己那一条 `--vllm-endpoint`，其余参数原样透传给 `speculators.train`。

同一个 wrapper 也按 `LOCAL_RANK` 注入 `--mooncake-master`：每个 rank 只连**自己那台引擎所属的 master**（rank r 用 `ENGINES` 第 r 条那台机器的 master）。每个 rank 只跟一个引擎通信（`routes[LOCAL_RANK % 8]`），所以 rank ↔ 机器 ↔ master 是一一对应的，训练侧不需要同时连两个 master，也没有任何共享文件被改。`HOST_MASTERS` 不填就退回单 master（`MOONCAKE_MASTER`，默认 `127.0.0.1:50051`）；master 在别的机器上用 `MOONCAKE_MASTER=<IP>:50051` 覆盖。

### 8.2 关键参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--speculator-type` / `--draft-arch` / `--draft-hidden-act` | `dspark` / `qwen3` / `silu` | 草稿类型与结构（DSpark 强制） |
| `--num-layers` | `5` | 草稿自己的 decoder 层数 |
| `--block-size` / `--max-anchors` | `8` / `512` | DFlash 块长 / anchor 上限 |
| `--target-layer-ids` | `5 23 42 60 78` | 须与引擎侧 `TARGET_LAYER_IDS` 逐位一致；5 项全进 fc 由基线 commit 自带的上游 PR #1224 提供，无需环境变量 |
| `--full-attention-indices` | `0 1 2 3 4` | 需要显式传，否则草稿使用 sliding_attention |
| `--optimizer` / `--lr` | `adamw` / `3e-4` | 需要显式传 optimizer，默认是 `muon` |
| `--scheduler-type` | `cosine` | 需要显式传，默认 `linear` |
| `--epochs` / `--checkpoint-freq` | `5` / `0.1` | 5 轮；每 10% 存一次 |
| `--total-seq-len` | `8192` | 与数据 prepare 的 `--seq-length` 一致 |
| `--loss-fn` | `{"ce": 0.1, "tv": 0.9}` | TV 主导 |
| `--markov-rank` / `--markov-head-type` | `256` / `vanilla` | Markov 低秩 logit-bias 头 |
| `--enable-confidence-head` / `--confidence-head-with-markov` | 开 | Confidence Head + Markov 联合 |
| `--num-workers` | **`12`** | dataloader worker 数；每个 worker 建一个 mooncake store，host 内存实测条件见 §10。**不是越多越好**：mooncake 通路上真正卡住的是库内硬编码的 60 秒单笔传输上限（§10），worker 越多、同时在飞的 hidden-states 越多、单笔排队越久，越过 60 秒就整批失败并重试。`12` 已验证可稳定跑（280 步、无硬超时）；`24` / `32` 会撞上这个上限（`32` 在租约修好后仍然 0 步；且 8 进程 × 32 worker = 256 个 worker 进程时 host 内存耗尽，见 §10） |
| `--hidden-states-backend` | `mooncake` | 引擎把 hidden-states 写进 mooncake store，训练器按 master 地址取，用完即删 |
| `--mooncake-master` / `--mooncake-metadata-server` / `--mooncake-protocol` | `127.0.0.1:50051` / `P2PHANDSHAKE` / `tcp` | master 地址须与引擎侧 `MOONCAKE_MASTER` 一致 |
| `--mooncake-global-segment-gib` / `--mooncake-local-buffer-gib` | `0` / `1` | 训练侧每个 worker 的 store 内存 / 本地暂存。`0`＝worker **不向 master 挂 segment**：置 0 后池子里只剩引擎的 segment。训练侧只读不写，不挂 segment 不影响取数 |
| `ENGINES`（config） | 8 个 `<IP>:<端口>` | 端点表；`route_rank_endpoints_glm5_2.py` 归一化为 URL 后按 `LOCAL_RANK` 取第 r 条注入 `--vllm-endpoint` |
| `--on-missing` | `generate` | 训练侧本地没有 hidden-states 时在线生成，取回后框架自动删除（`--on-generate` 上游已 deprecated：默认即 `delete` 且仅此一值，显式传只会收到 FutureWarning，两个样例的上游基线行为一致、均不传） |

训练产出的 checkpoint **不能直接发布**：verifier 是 QuaRot 量化模型，发布前必须先做 §9 的 `fc` 反向旋转。

## 9. 发布前：对 `fc` 反向旋转（de-QuaRot）

主模型是 W4A8 / W8A8 量化模型的 DSpark 草稿，发布前都要做这一步。

本样例的 verifier 是 QuaRot 量化模型，残差流与各层 hidden-states 都处在旋转后的坐标系，所以草稿的 `fc` 从初始化起就在旋转坐标系里。

引擎加载草稿时假定**草稿未旋转**，会自动把 `fc.weight` 按 `hidden_size` 分块右乘 `Q`（`vllm_ascend/models/qwen3_dspark.py` 的 `process_weight()` / `patch_draft_quarot.py`）。所以训练好的草稿在发布前需要先反向旋转一次：

```bash
python utils/derotate_draft_weights.py \
    --draft  ./output/dspark_glm52_800k/checkpoints/<要发布的 checkpoint> \
    --target /data/GLM-5.2-w4a8c8 \
    --output /path/to/publish/GLM-5.2-DSpark-derotated
```

checkpoint 由 `save_pretrained` 写出（`model.safetensors` + `config.json` + `config.py`，存在 `generation_config.json` 时脚本会一并带上）；不含 tokenizer——草稿与 verifier 共享词表，部署侧 tokenizer 默认取 verifier 的。

## 10. 已知问题与注意事项

- **训练机设了 HTTP 代理时要先清掉**：训练进程要直连 8 个 HS 引擎端点（`<引擎IP>:6700-6703`）和 mooncake master（`50051`），走代理会取不到 hidden-states。启动脚本只负责拉起训练，这一步要自己先做：

  ```bash
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  ```

  不方便清环境变量时，也可以把这些地址加进 `no_proxy`。

- **每个引擎的握手端口要唯一**：`MC_HANDSHAKE_PORT_BASE` 起每个引擎 +1，默认 51010-51013。同机多引擎撞端口会起不来，master 也要能连到这些端口。

- **训练侧 host 内存（每个 dataloader worker 各建一个 mooncake store）**：store 占 `MOONCAKE_GLOBAL_SEGMENT_GIB`（向 master 挂的 segment，本样例置 0，语义见 §8.2）+ `MOONCAKE_LOCAL_BUFFER_GIB`（本地暂存）；worker 的 host 内存不止 store，dataloader 的 pin memory（torch_npu 经 `aclrtMallocHost` 锁页，hidden-states 等大张量都算在内）也在内。global 不为 0 时 worker 会在 master 登记可写入 segment，训练机可能被引擎选为 hidden-states 写入落点；训练侧只读不写，置 0 即可。实测条件：2 TB 内存训练机、8 训练进程 × 12 worker、mooncake global 0 / local 1 GiB，稳定跑完 280 步；同机 `--num-workers 32`（8 进程 × 32 worker = 256 个 worker 进程）时 host 内存耗尽，报 `OutOfMemoryError in pin memory thread` / `aclrtMallocHostWithCfg 207001`。

- **mooncake 单笔传输 60 秒硬上限（`--num-workers` 调小的原因）**：上限来自 mooncake 传输引擎内部 `TransferEngineOperationState::wait_for_completion()` 的等待超时（`mooncake-store/src/transfer_task.cpp`；0.3.9 / 0.3.10 源码为 `constexpr int64_t timeout_seconds = 60;`，本样例锁定的 `0.3.11.post1` 与未打补丁的 `0.3.13` 相同——编译期常量；失败时报 `Failed to complete transfers after 60000 milliseconds for batch`）。这是**等待端**的超时，与传输本身速率无关：worker 越多，同时在飞的 hidden-states 越多，单笔在队列里等到的时长越久，越过 60 秒即整批失败并重试。`MOONCAKE_TRANSFER_TIMEOUT` 等环境变量不作用于这条等待路径。
