# 基于 vLLM Speculators 的 Qwen3.8-27B DSpark 草稿模型 NPU 训练样例

## 1. 概述

本样例基于 [vllm-project/speculators](https://github.com/vllm-project/speculators) 训练框架，在昇腾 NPU 上在线训练 Qwen3.8-27B 的 **DSpark 草稿模型**（draft model）。草稿模型用于投机解码（speculative decoding）推理加速：推理时先由轻量草稿模型一次生成一个候选块（draft block），再由 verifier 模型单次前向并行验证，在不改变输出分布的前提下提升生成吞吐。DSpark 在 DFlash 基础上增加了建模块内 token 依赖的 Markov Head 与逐位置接受概率预测的 Confidence Head。

**评测结果**（投机解码平均接受长度 AL，每个 verifier 前向接受的平均 token 数，越高加速比越高）：

| 类别 | 数据集 | AL |
| --- | --- | --- |
| 数学 | GSM8K | 5.89 |
| 数学 | MATH500 | 4.61 |
| 代码 | HumanEval | 4.4 |
| 代码 | MBPP | 4.36 |
| 通用中文 | Magpie | 3.35 |
| 通用 | Alpaca | 3.22 |
| 通用 | MTBench | 3.4 |
| 长序列 | Longbench-v2 | 4.29 |
| Agent | 自研Agent数据集\* | 3.87 |

\* 自研 Agent 评测数据集（非公开）。

本样例基于官方 speculators 框架，hidden-states 传输用官方 backend 组合：**多机免共享存储用 mooncake backend**——vLLM 侧 connector 将 hidden-states 写入 mooncake 分布式 store，训练侧按 req_id 经传输引擎直连拉取（TCP/RDMA），彻底免去共享存储（如 cephfs）的跨机写依赖与其长尾抖动；**单机零依赖用 file backend**——safetensors + 文件锁落共享文件系统。多 vLLM endpoint 场景由启动脚本 `scripts/route_rank_endpoints_qwen3_8_27b.py` 按 rank 路由（分摊各 rank 的生成请求负载），无需改动训练框架。

训练得到的草稿模型 checkpoint 可用于 vLLM-Ascend 投机解码推理，benchmark 验证与部署对接 `cann-recipes-infer` 仓。

## 2. 使用的产品型号

| 项目 | 规格 |
| --- | --- |
| 产品型号 | Atlas A2 系列（Ascend 910B3） |
| 卡数 | 8 卡（4 卡 vLLM 提取服务 + 4 卡训练；亦支持双机每节点 2×TP2 服务 + 2 训练卡） |
| 操作系统 | Ubuntu 22.04.5 LTS（ARM64） |
| Python | 3.12 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post2（引擎与训练同一环境，实测版本，见 requirements.txt） |
| vLLM | **v0.26.0**（@ `568afb3a13806beb53bb2e6bd518269357b237c0`） |
| vllm-ascend | **v0.26.0rc1**（@ `f2f74a16c3c50a76f4349d807918e83edec1e35c`） |
| CANN / NPU 驱动 | CANN **9.1.0** / 固件 9.0.0.0.205 / 软件 26.0.rc1（npu-smi 26.0.rc1） |

## 3. 文件说明

| 文件/目录 | 说明 |
| --- | --- |
| `README.md` | 中文训练说明（本文件） |
| `scripts/serve_vllm_hs_qwen3.8_27b.sh` | 启动单个 vLLM hidden-states 提取服务（后台运行，mooncake/file backend 可选，含 NPU 环境变量与调优参数） |
| `scripts/vllm_regenerate_qwen3.8_27b.sh` | 启动 on-policy response regeneration 用 vLLM 服务（不启用 hidden-states 提取，含 `FULL_DECODE_ONLY`） |
| `scripts/route_rank_endpoints_qwen3_8_27b.py` | 训练启动 wrapper：按 `LOCAL_RANK` 从端点表（`ENGINES`）注入单一 `--vllm-endpoint` 后移交 `speculators.train`（多 endpoint 生成负载分摊，位于训练框架之外） |
| `scripts/train_config_qwen3.8_27b.yaml` | 训练配置：模型路径、端点表、取层、超参与本模型专用 flag（平铺 yaml，启动脚本按文件名自动读取，同名环境变量优先） |
| `scripts/train_dspark_qwen3.8_27b_online.sh` | 在线训练启动脚本（后台运行；按文件名自动 source 上述 config，torchrun 目标为上述 wrapper） |
| `utils/sample_opb_by_source.py` | 按 source 分层抽样英文源池（先抽 65w 再重生成，保持各子集原始比例，§6.2.2） |
| `../utils/select_cmb_chinese_prompts.py` | 从 CMB-Eagle-Mix-330k 筛中文 prompt（speculators/utils/ 共用脚本，口径见 §6.2.3） |
| `requirements.txt` | Python 依赖（锁定版本） |

## 4. 源码准备

```bash
# 1. 克隆原仓并固定到验证过的基线 commit（或更新的 commit）
git clone https://github.com/vllm-project/speculators.git
cd speculators
git checkout e8f8879
#    默认取层 1 16 31 46 61 不含末层（id 64），该基线即可运行本样例；
#    若把 --target-layer-ids 改为含末层，需要上游 PR #1224（修复「取层含末层时的
#    fc 输入布局」，已合入上游 main、squash 为 d3c3306）。在 e8f8879 上 cherry-pick
#    该 PR 会在 tests/unit/train/test_data.py 冲突，建议直接 checkout d3c3306 或
#    更新的 commit（超出本样例验证基线）

# 2. 可编辑安装（先子包 hs_connectors，再本体；vLLM 服务侧环境同样需要 hs_connectors）
cd hs_connectors && pip install -e . && cd ..
pip install -e .

# 3. mooncake backend 传输引擎（Ascend 构建；vLLM 服务侧环境同装。file backend 可跳过）
pip install mooncake-transfer-engine-npu==0.3.11.post1

# 4. 将样例启动脚本拷入仓库 scripts/，数据工具拷入 utils/
SAMPLE=../cann-recipes-train/llm_spec/speculators/qwen3.8_27b
cp $SAMPLE/scripts/*.sh $SAMPLE/scripts/*.py scripts/
mkdir -p utils
cp $SAMPLE/utils/*.py utils/
cp $SAMPLE/../utils/endpoint_routes.py utils/
```

## 5. 环境准备

1. NPU 驱动与 CANN toolkit。裸机部署需先 source CANN 环境（vllm-ascend 依赖 pyACL 的 `acl` 模块，未 source 时 `Engine core initialization failed: No module named 'acl'`）：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   `serve_vllm_hs_qwen3.8_27b.sh` 已内置该 source（幂等，可重复执行）。
2. 安装训练侧依赖（位于样例目录 `llm_spec/speculators/qwen3.8_27b/`）：

   ```bash
   pip install -r ../cann-recipes-train/llm_spec/speculators/qwen3.8_27b/requirements.txt
   ```

3. 关键版本组合：

   | 组件 | 版本要求 | 说明 |
   | --- | --- | --- |
   | vllm-ascend | **v0.26.0rc1**（@ `f2f74a16c3c50a76f4349d807918e83edec1e35c`） | hybrid verifier 并发 hidden-states 提取的 NaN 修复已在该版本合入上游（见 §9），无需任何 patch |
   | vLLM | **v0.26.0**（@ `568afb3a13806beb53bb2e6bd518269357b237c0`） | **注：v0.31.0 收的 abort 安全修复 [vllm#56841](https://github.com/vllm-project/vllm/pull/56841) 在 0.26.0 不存在，训练器超时 abort 在途 KV transfer 可能 assert 拉垮 EngineCore——本样例若改用 ≥ v0.31.0 更稳妥** |
   | CANN | **9.1.0** | 硬门槛：低于 9.1.0 未在本样例验证 |
   | torch / torch_npu | 2.10.0+cpu / 2.10.0.post2（requirements.txt 锁定，已验证；引擎与训练同一环境） | CPU wheel，NPU 算力由 torch_npu 提供 |
   | transformers | 5.14.1 | — |
   | mooncake-transfer-engine-npu | `0.3.11.post1`（Ascend 构建） | mooncake backend 传输引擎；vLLM 与训练两侧环境同装同版本，file backend 无需。自带 `mooncake_master`，启动时**必须**带 `--default_kv_lease_ttl 300000`（默认 5 秒租约撑不住单笔 hidden-states，见 §7） |

## 6. 数据准备

数据准备流程全部在已 `pip install -e .` 的 speculators 仓库根目录执行。两套配方：§6.1 是公开数据集快速验证（off-policy），§6.2 是本样例主推的 on-policy 重生成管线（源池 1.42M，实际训练取 65w 英文 + 15w 中文 = 80w）。

### 6.1 快速验证（ultrachat / sharegpt）

公开数据集走 `prepare-data` 时仍需要 vLLM 服务来派生 loss mask；off-policy 配方指的是“原 assistant 回答直接作为监督目标、不再重生成”，但 `prepare-data` 本身仍需 `--render-endpoint`。

```bash
# 0. 启动 verifier vLLM 服务（与 §6.2.1 同一份命令即可）
MODEL_PATH=/path/to/Qwen3.8-27B bash scripts/vllm_regenerate_qwen3.8_27b.sh

# 1. 准备数据
python -m speculators prepare-data \
    --model /path/to/Qwen3.8-27B \
    --render-endpoint http://127.0.0.1:8000 \
    --data ultrachat \
    --output ./output/dspark_qwen3_8_27b_ultrachat \
    --max-samples 10000 \
    --seq-length 8192
```

> 冒烟常用 `sharegpt`；预设注册清单见 `src/speculators/data_generation/configs.py` 的 `DATASET_CONFIGS`。
> §6.2.4 那种已经含 `input_ids` + `loss_mask` 列的 JSONL（regen 的产出）才会被 `prepare-data` 走 passthrough，**不需要** `--render-endpoint`。

### 6.2 主推配方（open-perfectblend + 中文 thinking，80w）

英文源数据集 [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)（1.42M 条，多子集合并）。**on-policy 重生成** 共 5 步（启动 regen 服务 → 按 source 拆分源池 → 分层抽样英文 65w → 重生成英文 65w 与中文 CMB 数据 → 混合 prepare），训练消费 `./output/dspark_qwen3_8_27b_mix80w/`。

#### 6.2.1 启动 regen 用 vLLM 服务

```bash
MODEL_PATH=/path/to/Qwen3.8-27B bash scripts/vllm_regenerate_qwen3.8_27b.sh
```

脚本默认使用 NPU 4,5、端口 8000，可通过 `ASCEND_RT_VISIBLE_DEVICES` 和 `PORT` 覆盖。

#### 6.2.2 英文：按 source 分层抽样 65w

先按 `source` 把 open-perfectblend 拆成每 source 一个输入文件：

```bash
python3 - <<'EOF'
from datasets import load_dataset
import json
import os

os.makedirs("./data/opb_by_source", exist_ok=True)
fhs = {}
for row in load_dataset("mlabonne/open-perfectblend", split="train", streaming=True):
    src = row.get("source") or "unknown"
    fh = fhs.get(src)
    if fh is None:
        fh = fhs[src] = open(f"./data/opb_by_source/{src}.jsonl", "w", encoding="utf-8")
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
for fh in fhs.values():
    fh.close()
print(sorted(fhs))
EOF
```

再分层抽 65w：

```bash
python utils/sample_opb_by_source.py \
    --src-dir ./data/opb_by_source \
    --out ./data/opb_650k.jsonl \
    --num-samples 650000 \
    --seed 42
```

脚本先统计每个 source 的条数，再按比例把 65w 拆成整数（largest-remainder 分配，保证各 source 的占比与源池一致、整数之和恰为 65w），最后在每个 source 内部随机抽；分片文件按文件名归到 source（`xxx_s0of10.jsonl` 与 `xxx_s1of10.jsonl` 属于同一个 source）。

#### 6.2.3 on-policy 重生成（英文 65w 与中文 CMB 数据）

英文（输入就是上一步抽好的 65w）：

```bash
python -m speculators regenerate-responses \
    --dataset ./data/opb_650k.jsonl \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model /path/to/Qwen3.8-27B \
    --outfile ./output/opb_650k_regen.jsonl \
    --resume \
    --concurrency 64 \
    --max-tokens 8192 \
    --max-retries 3
```

中文数据取自 [CMBTech/CMB-Eagle-Mix-330k](https://www.modelscope.cn/datasets/CMBTech/CMB-Eagle-Mix-330k)（ModelScope，Apache-2.0，33w 条中英混合，其中文部分即 swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT 蒸馏数据），用 `../utils/select_cmb_chinese_prompts.py`（speculators/utils/ 共用脚本）按**「中文占比 >= 0.24、分母=总字符数」降序取前 10w**（每条只取第一个 human 轮作 prompt）。

```bash
pip install modelscope
modelscope download --dataset CMBTech/CMB-Eagle-Mix-330k --local_dir ./data/cmb

# 筛出中文 prompt（只取首个 human 轮；输出每行一个 {"prompt": ...}）
python ../utils/select_cmb_chinese_prompts.py \
    --src ./data/cmb/cmb_eagle_mix_330k.jsonl \
    --out ./data/cmb_zh_prompt_100k.jsonl \
    --min-zh-ratio 0.24 \
    --num-prompts 100000

python -m speculators regenerate-responses \
    --dataset ./data/cmb_zh_prompt_100k.jsonl \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model /path/to/Qwen3.8-27B \
    --outfile ./output/cmb_zh_regen.jsonl \
    --resume \
    --concurrency 64 \
    --max-tokens 8192 \
    --max-retries 3
```

`--dataset` 传选取脚本的产物即可（非注册预设时按本地文件加载，`prompt` 字段被包装成单轮 user 消息重新生成，原 assistant 轮不进入重生成）。

两次重生成（英文、中文）的要点：

- `--dataset` 传本地 .jsonl 即可（非注册预设时按本地文件加载，`conversations` / `messages` 都能直接识别，行里的 `source` 等额外字段不影响）。
- `--endpoint` 指向上述服务；`--model` 使用模型路径，与服务暴露的模型 ID 一致，同时供脚本加载 tokenizer。
- 英文 65w + 中文 10w 共 75w 条，单 endpoint 重生成约数小时。
- `--resume` 必传：输出为追加写，重跑靠 `primary_id` 跳过已完成对话，缺了会重复。

中文最终取约 15w 行，比筛选后的源池（10w 条）多出来的部分用「复制加权」补齐：全量写一遍，再按固定种子随机重复其中约 5w 条（这批样本在训练里出现两次，等价于对它们加权 2×）：

```bash
python3 - <<'EOF'
import random

src = "./output/cmb_zh_regen.jsonl"
dst = "./output/cmb_zh_regen_15w.jsonl"
rows = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
dup = min(50000, len(rows))
extra = random.Random(42).sample(rows, dup)
with open(dst, "w", encoding="utf-8") as f:
    for l in rows + extra:
        f.write(l + "\n")
print(f"{len(rows)} + {dup} = {len(rows) + dup} 行 -> {dst}")
EOF
```

加权重落在单个文件里，下一节 prepare 一次传入即可。也可以不合成一个文件，改成「全量文件 + 复制出来的 5w 文件」两个 `--data` 传入，结果相同（`prepare-data` 的 `--data` 可重复传，多文件按序拼接）。

#### 6.2.4 训练数据 prepare

混合 prepare（英文重生成产物 65w + 中文加权 15w = 80w）：

```bash
python -m speculators prepare-data \
    --model /path/to/Qwen3.8-27B \
    --data ./output/opb_650k_regen.jsonl \
    --data ./output/cmb_zh_regen_15w.jsonl \
    --output ./output/dspark_qwen3_8_27b_mix80w \
    --seq-length 8192
```

`prepare-data` 识别到 input_ids/loss_mask 列后走 passthrough 直接整理为 Arrow，无需手工清洗；`--data` 可传多次，多文件按序拼接混合。`--seq-length` 须与 §8 训练 `--total-seq-len` 一致。

## 7. 启动 mooncake master 与 vLLM 提取服务

以单机 8 卡、双提取实例为例（实例 A 占卡 0,1、端口 8010；实例 B 占卡 2,3、端口 8011）：

```bash
# 0.（mooncake backend 前置，每集群一次）任一节点常驻 master；
#    各节点须已装 mooncake-transfer-engine-npu（§4 步骤 3），网络放通
#    各节点->master 与 master->各节点握手端口；
#    必须带 --default_kv_lease_ttl：connector 侧不设 TTL，对象寿命完全由
#    master 租约决定，默认 5 秒租约撑不住单笔 hidden-states 传输，
#    租约内未被训练侧取走会被回收导致取数失败
mooncake_master --rpc_port=50051 --default_kv_lease_ttl 300000 &

# 实例 A（MOONCAKE_LOCAL_HOSTNAME 必须是本机可路由 IP；同机多实例时
# MC_HANDSHAKE_PORT 每实例唯一）
PORT=8010 DEVICES=0,1 MODEL_PATH=/path/to/Qwen3.8-27B \
MOONCAKE_LOCAL_HOSTNAME=<本机IP> MC_HANDSHAKE_PORT=51010 \
    bash scripts/serve_vllm_hs_qwen3.8_27b.sh
# 实例 B（后台拉起，无需另开终端）
PORT=8011 DEVICES=2,3 MODEL_PATH=/path/to/Qwen3.8-27B \
MOONCAKE_LOCAL_HOSTNAME=<本机IP> MC_HANDSHAKE_PORT=51011 \
    bash scripts/serve_vllm_hs_qwen3.8_27b.sh
```

两个实例各自后台运行：日志 `./serve_vllm_hs_<端口>.log`、PID `./serve_vllm_hs_<端口>.pid`，停止 `kill $(cat serve_vllm_hs_<端口>.pid)`；是否就绪以日志为准。

单机零依赖变体：`HS_BACKEND=file PORT=8010 DEVICES=0,1 MODEL_PATH=/path/to/Qwen3.8-27B bash scripts/serve_vllm_hs_qwen3.8_27b.sh`——无 master、无引擎依赖；hidden-states 写 `HS_PATH`（默认 `/dev/shm/hs_<端口>`），训练侧与 vLLM 须共享文件系统（容量账约 61KB/token，tmpfs 满前先清空）。

> 脚本已内置 `--enable-chunked-prefill` + `max-num-batched-tokens 32768`：较默认配置提取吞吐约 +25~35%，输出 token_ids 不变。

## 8. 训练

```bash
bash scripts/train_dspark_qwen3.8_27b_online.sh
```

脚本后台运行：日志 `$OUTPUT_DIR/train.log`、PID `$OUTPUT_DIR/train.pid`，停止 `kill $(cat $OUTPUT_DIR/train.pid)`。

启动脚本按自身文件名读取同目录 `train_config_qwen3.8_27b.yaml`（平铺 yaml，同名环境变量优先于文件值），`CONFIG=<路径>` 可指定其他配置。

关键配方（写在 `train_config_qwen3.8_27b.yaml`，详见其注释）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--speculator-type` | `dspark` | 草稿类型 |
| `--num-layers` / `--draft-arch` | 5 / `qwen3` | 草稿深度与结构 |
| `--block-size` | 8 | 草稿块长度 |
| `--max-anchors` | 512 | anchor 上限 |
| `--target-layer-ids` | 1 16 31 46 61 | 提取的 verifier 层，须与 vLLM 侧一致（末层 64 由 launch_vllm 自动追加） |
| `--full-attention-indices` | 0 1 2 3 4 | 草稿各层全注意力（0 .. num_layers-1）。训练↔推理对齐关键：推理时 block 内恒为 non-causal 且全上下文可见，不传时训练默认全层 sliding window，与推理双重不一致，须改完从头重训 |
| `--enable-confidence-head --confidence-head-with-markov` | — | Confidence Head + Markov 联合 |
| `--loss-fn` | `{"ce": 0.1, "tv": 0.9}` | CE + TV 混合损失 |
| `--optimizer` | `adamw` | 上游默认为 `muon`，本样例主配方用 `adamw` |
| `--total-seq-len` | 8192 | 训练序列长度（serving 侧 `--max-model-len` 9216 = 8192 + 1024 余量） |
| `--lr` / `--epochs` | 6e-4 / 5 | 优化器 |
| `--on-missing` / `--on-generate` | `generate` / `delete` | 在线生成、用后即删 |
| `--hidden-states-backend` | `mooncake` | 官方 backend：多机免共享存储；单机零依赖可 `file`（与 serve 脚本一致） |
| `--mooncake-master` / `--mooncake-protocol` / `--mooncake-global-segment-gib` / `--mooncake-local-buffer-gib` | `127.0.0.1:50051` / `tcp` / `1` / `1`（train_config 默认值） | master 须与 vLLM 侧一致；8k 样本 HS ≈ 0.5GB/条，local buffer 1 GiB 余量充足 |
| `--vllm-endpoint` | wrapper 注入 | 由 `route_rank_endpoints_qwen3_8_27b.py` 按 rank 从端点表（`ENGINES`）注入，CLI 不直接出现 |
| `--num-workers` | 24 | dataloader worker 数（均消费本 rank 绑定的 endpoint） |

双 endpoint 路由（默认已配）：`ENGINES="127.0.0.1:8010 127.0.0.1:8011"`，rank 0/2 → 实例 A、rank 1/3 → 实例 B，分摊各 rank 的生成请求负载（mooncake 下 hidden-states 数据腿走 store，与 endpoint 绑定无关）。

训练监控：tensorboard 日志（`--log-dir` 指向的目录）。

## 9. 已知问题与注意事项

- **mooncake backend 不支持按索引缓存复用**：`MooncakeTransfer.get_cached` 恒为空，只走在线生成流程（`--on-missing generate` + `--on-generate delete`）；需要缓存/复用数据集时用 `--hidden-states-backend file`。
- **file backend 下 vLLM 侧 safetensors 建议 ≥ 0.8.0**：`<0.8.0` 的 `save_file` 持有 GIL，会串行化 connector 的写入线程池（训练侧读不受影响；mooncake 不落盘亦不受影响；requirements 已锁 0.8.0）。
- **端点表 `ENGINES` 为空格分隔的 `host:port` 列表**：rank r 使用第 r % 条数 个 endpoint，分摊各 rank 的生成请求负载；mooncake 下 hidden-states 数据腿经 store 传输、与 endpoint 绑定无关，file backend 下训练侧须与所选实例共享文件系统。
- **mooncake backend 注意事项**：① vLLM 与训练两侧进程都必须设置 `MOONCAKE_LOCAL_HOSTNAME=<本机可路由 IP>`——不设会注册成 127.0.0.1，master 侧出现幽灵 segment，跨机传输报 `TRANSFER_FAIL(-800)`（两个启动脚本未设时会直接报错退出）；② 同机多 vLLM 实例时 `MC_HANDSHAKE_PORT` 每实例必须唯一，否则引擎互抢端口；③ 训练侧每个 dataloader worker 各建一个 store（global segment + local buffer，本样例各 1 GiB），host 内存按 worker 进程总数消耗；④ master 为单点，须常驻并保证各节点网络可达；⑤ 启动时 cwd 下若有 `vllm/` 目录（源码仓）会遮蔽已安装的 vllm 包，统一在 speculators 仓库根目录启动。
- **训练 dataloader 的 worker 固定以 spawn 方式启动**（上游写死，CANN/ACL 非 fork 安全）：每个 worker 为全新进程，首轮取数会明显慢于后续（如 4 rank × 24 worker = 96 进程），属预期现象，仅在启动时发生一次。
- **w8a8 量化 verifier 的 hidden-states 有动态量化漂移**（以后层为主），token_ids 完全一致；对草稿训练收敛无实质影响，但做 HS 数值对齐分析时需知悉。
- **hybrid（GDN 线性注意力 + 全注意力混合）verifier 并发 hidden-states 提取的 NaN 已在 vllm-ascend v0.26.0rc1 上游修复，无需补丁**：旧版中 hidden-state 层与 attn-mamba 组共用物理页，并发下写越界会污染 hidden-states（公开 Qwen3 dense/MoE 模型不涉及）。按 §5 版本组合部署即可。
