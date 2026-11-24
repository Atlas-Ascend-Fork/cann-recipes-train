# DeepSeek-R1 RL训练优化实践样例

## 概述

本样例针对DeepSeek-R1模型，基于[verl开源框架](https://github.com/volcengine/verl)，配合MindSpeed-LLM和vLLM-Ascend框架，完成RL训练全流程的优化适配。优化点介绍可参见[基于veRL前端&A3集群的DeepSeekR1模型RL训练优化实践](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_rl/deepseek/zh/deepseek_rl_train_optimization.md)。

本样例基于Atlas A3 128卡集群，加载真实权重，使用deepscaler数据集，Prefill/Decode阶段长度分别为1K与3K，系统吞吐可达到120TPS/卡。随着训练的进行，模型response_length会有所增长，系统吞吐可以进一步提升。
| 基础模型             | 机器型号     | GBS | n_samples | step | max_prompt_length | max_response_length | 端到端TPS |
|---------------------|----------|-----|-----------|--------- | ----------|------------|---------|
| DeepSeek-V3-671B    | Atlas A3 128卡 | 512 | 16        | 2 |  1024              | 3072       | 120     |

> 性能更新说明：以上为训练侧重计算4层的性能。为确保加载权重后长稳训练，训练重计算改为8层，训练13步左右后，吞吐也可从初始112TPS/卡提升至120TPS/卡。

## 硬件要求
产品型号：Atlas A3 系列

最少卡数：128张A3

## 目录结构说明
```bash
├─megatron_patches            # megatron库相关patch目录
│  └─core                     # 对应megatron/core目录
│      └─tensor_parallel      # 对应megatron/core/tensor_parallel目录，其中有相关patch代码
├─mindspeed_patches           # mindspeed库相关patch目录
│  └─core                     # 对应mindspeed/core目录
│      └─tensor_parallel      # 对应mindspeed/core/tensor_parallel目录，其中有相关patch代码
├─verl_patches                # verl库相关patch目录
│  ├─features                 # 重要独立特性目录
│  │  └─rollout_optimize      # 推理负载均衡优化特性目录
│  ├─models                   # 对应verl/models目录，其中有相关patch代码
│  │  └─mcore                 # 对应verl/models/mcore目录，其中有相关patch代码
│  ├─scripts                  # 重要脚本目录，包含训练启动、数据文件转化、权重转换等脚本
│  │  └─internal              # 训练启动脚本所调用的子脚本目录，其中可配置verl训练的重要参数
│  ├─single_controller        # 对应verl/single_controller目录，其中有相关patch代码
│  │  └─base                  # 对应verl/single_controller/base目录，其中有相关patch代码
│  │      └─megatron          # 对应verl/single_controller/base/megatron目录，其中有相关patch代码
│  ├─trainer                  # 对应verl/trainer目录，其中有相关patch代码
│  │  ├─config                # 对应verl/trainer/config目录，其中有相关patch代码
│  │  └─ppo                   # 对应verl/trainer/ppo目录，其中有相关patch代码
│  ├─train_engine             # 训练引擎相关的重要patch代码目录
│  ├─utils                    # 对应verl/utils目录，其中有相关patch代码
│  │  ├─megatron              # 对应verl/utils/megatron目录，其中有相关patch代码
│  │  └─reward                # 对应verl/utils/reward_score目录，其中有相关patch代码
│  └─workers                  # 对应verl/workers目录，其中有相关patch代码
│      ├─actor                # 对应verl/workers/actor目录，其中有相关patch代码
│      ├─sharding_manager     # 对应verl/workers/sharding_manager目录，其中有相关patch代码
│      └─vllm_rollout         # 对应verl/workers/rollout/vllm_rollout目录，其中有相关patch代码
└─vllm_ascend_patches         # vllm_ascend相关patch代码目录
```

## 基于Dockerfile构建环境
> 环境搭建可以基于Dockerfile快速实现，我们已经在Dockerfile里配置了必要的昇腾软件和其他第三方软件的依赖。如果遇到网络不通等问题，也可以参考附录中的[手动准备环境](#手动准备环境)章节。

1. 基于Dockerfile创建镜像。
   ```bash
   # 请预先下载本样例提供的Dockerfile文件：Dockerfile.vllm_ascend.mindspeed.deepseekV3
   # 随后基于该Dockerfile创建docker image。请传入镜像名称，例如 your_image_name
   docker build -t your_image_name -f Dockerfile.vllm_ascend.mindspeed.deepseekV3 .

   # 请设置容器名称，例如 your_docker_name，镜像名称同上一步
   container_name=your_docker_name
   image_name=your_image_name:latest

   # 执行docker run命令创建容器，可通过-v按需挂载宿主机目录至容器
   docker run -itd \
   --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
   -v /usr/local/dcmi:/usr/local/dcmi \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
   -v /usr/local/sbin/:/usr/local/sbin/ \
   -v /data/:/data/ \
   -v /home/:/home/ \
   -v /etc/localtime:/etc/localtime \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /dev/shm:/dev/shm \
   --net=host \
   --name ${container_name} \
   --privileged ${image_name} /bin/bash

   # 执行docker exec命令进入容器
   docker exec -it -u root ${container_name} bash
   ```

2. 源码准备。
   ```bash
   # 下载本样例所在代码仓，以master分支为例
   git clone https://gitcode.com/cann/cann-recipes-train.git

   # 添加镜像中已经准备好的依赖文件
   cp -r /workspace/verl/verl ./cann-recipes-train/llm_rl/deepseek/
   cp -r /workspace/vllm/vllm ./cann-recipes-train/llm_rl/deepseek/
   cp -r /workspace/vllm-ascend/vllm_ascend ./cann-recipes-train/llm_rl/deepseek/
   cp -r /workspace/Megatron-LM/megatron ./cann-recipes-train/llm_rl/deepseek/
   cp -r /workspace/MindSpeed/mindspeed ./cann-recipes-train/llm_rl/deepseek/
   cp -r /workspace/MindSpeed-LLM/mindspeed_llm ./cann-recipes-train/llm_rl/deepseek/
   cd ./cann-recipes-train/llm_rl/deepseek/   # 回到本样例目录

   # 安装依赖的python库
   pip install -r requirements.txt
   ```

3. 修改verl源码，使能patch修改。

   在 `verl/workers/megatron_workers.py: line21` 处插入以下代码：
   ```python
   from verl_patches import prelude_patch
   ```

## 数据集准备
本样例使用的deepscaler数据集准备方法如下：
```bash
# 在本样例目录下创建deepscaler数据集目录
mkdir -p ./data/deepscaler
cd ./data/deepscaler

# 从魔塔社区下载deepscaler数据集的json文件
pip install modelscope
modelscope download --dataset agentica-org/DeepScaleR-Preview-Dataset deepscaler.json --local_dir .

# 使用本样例提供的工具脚本，将deepscaler.json文件转为train.parquet和test.parquet两个文件并存放至数据集目录
cd ../..
python ./verl_patches/scripts/json_to_parquet.py --output_dir ./data/deepscaler --json_path ./data/deepscaler/deepscaler.json
```

gsm8k等其他数据集准备方法可参考[verl官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)。

## 模型权重准备
本样例使用的DeepSeek-V3模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放至样例的./DeepSeek-V3目录下（不加载权重实验也需要执行这步操作）
mkdir ./DeepSeek-V3
pip install modelscope
modelscope download --model deepseek-ai/DeepSeek-V3 configuration_deepseek.py tokenizer.json tokenizer_config.json --local_dir ./DeepSeek-V3
cp ./verl_patches/config.json ./DeepSeek-V3/config.json  # 使用本样例提供的config，去掉了量化与MTP

# 下载DeepSeek-V3完整FP8权重至指定目录，例如 your_fp8_weights（此步骤需要目录所在磁盘有650GB以上空间）
modelscope download --model deepseek-ai/DeepSeek-V3 --local_dir your_fp8_weights

# 将FP8权重转换为BF16权重并保存至指定目录，例如 your_bf16_weights（此步骤需要目录所在磁盘有1300GB以上空间）
python verl_patches/scripts/fp8_cast_bf16.py --input-fp8-hf-path your_fp8_weights --output-bf16-hf-path your_bf16_weights

# 将权重之外的文件复制到BF16权重目录下
cd your_fp8_weights
cp -r config.json configuration.json configuration_deepseek.py generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json your_bf16_weights
cd -  # 回到本样例根目录

# source环境变量，根据实际CANN安装目录调整
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 执行权重切分，切分后权重保存至指定目录，例如 your_sharded_weights（此步骤需要目录所在磁盘有1300GB以上空间）
pip install safetensors bitsandbytes
python verl_patches/scripts/convert_ckpt_deepseek3.py \
    --moe-grouped-gemm \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 8 \
    --num-layer-list 7,7,8,8,8,8,8,7 \
    --target-expert-parallel-size 8 \
    --load-dir your_bf16_weights \
    --save-dir your_sharded_weights \
    --moe-tp-extend-ep \
    --first-k-dense-replace 3 \
    --num-layers 61
```

## RL后训练执行

在本样例代码根目录下启动DeepSeekV3的RL后训练。

```bash
# 请注意，以下bash启动脚本中的内容需要手动配置
# source脚本路径：  根据实际CANN安装目录调整
# MASTER_ADDR：    ray集群主节点的IP地址，每个节点的脚本配置一致
# SOCKET_IFNAME：  集群中各节点自己的网卡名，可通过ifconfig命令查看
# LD_PRELOAD：     其中jemalloc文件位置需根据实际情况调整

# 基于随机权重的训练脚本
bash ./verl_patches/scripts/train_deepseekv3_256die_random_init.sh

# 基于真实权重的训练脚本（需要切分权重存储路径`your_sharded_weights`在集群内共享）
# 请注意，bash启动脚本中的`DIST_CKPT_PATH`环境变量需手动设置为切分权重的保存路径 your_sharded_weights
bash ./verl_patches/scripts/train_deepseekv3_256die_true_weight.sh
```

## 附录
### 手动准备环境

1. 创建vLLM-Ascend镜像。

   ```bash
   # 镜像下载
   docker pull quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler

   # 执行以下脚本创建容器，请传入容器名称，如 your_docker_name
   docker run -itd \
   --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
   -v /usr/local/dcmi:/usr/local/dcmi \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
   -v /usr/local/sbin/:/usr/local/sbin/ \
   -v /data/:/data/ \
   -v /home/:/home/ \
   -v /etc/localtime:/etc/localtime \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /dev/shm:/dev/shm \
   --net=host \
   --name your_docker_name \
   --privileged quay.io/ascend/vllm-ascend:v0.9.1-dev-openeuler /bin/bash

   # 执行docker exec命令进入容器
   docker exec -it -u root your_docker_name bash

   # 安装依赖软件
   yum install -y patch # openEuler系统
   apt install -y patch # Ubuntu系统
   ```
2. 在容器中安装CANN软件包与Ascend Extension for PyTorch软件包。
   - **CANN：8.2.RC1**

      请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载如下软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)进行安装。

      - 开发套件包：`Ascend-cann-toolkit_${version}_linux-${arch}.run`
      - 二进制算子包：`Atlas-A3-cann-kernels_${version}_linux-${arch}.run`
      - NNAL加速包：`Ascend-cann-nnal_${version}_linux-${arch}.run`

      软件包文件名中 `${version}` 表示CANN包版本号，`${arch}` 表示CPU架构（如aarch64、x86_64）。

   - **Ascend Extension for PyTorch：7.1.0**

      Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`7.1.0`，PyTorch版本为`2.5.1`。

      请参考[Ascend Extension for PyTorch安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html#ZH-CN_TOPIC_0000002389661329__zh-cn_topic_0000001800921750_zh-cn_topic_0000001731730474_section1945921143716)安装相应版本的torch_npu插件。

   - **Apex**

      本样例需要安装apex库，请参考[apex](https://gitcode.com/ascend/apex)构建安装。

   - **Jemalloc**

      本样例需要安装jemalloc库，请参考[高性能内存库jemalloc安装](https://gitcode.com/Ascend/MindSpeed-RL/blob/2.1.0/docs/install_guide.md#%E9%AB%98%E6%80%A7%E8%83%BD%E5%86%85%E5%AD%98%E5%BA%93-jemalloc-%E5%AE%89%E8%A3%85)。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-train.git

    # 安装依赖的python库
    cd cann-recipes-train/llm_rl/deepseek
    pip install -r requirements.txt
    ```

4. 下载依赖的开源框架代码。

   为了让使用者和开发者直观了解我们基于开源代码做的修改，本样例中只包含patch代码，其他框架代码需要拉取。

   返回cann-recipes-train项目代码上级目录，即执行git clone命令时所在目录，并执行如下命令，需注意，请确保环境能够正常连通网络。
   ```bash
   # 返回cann-recipes-train项目代码上级目录
   cd ../../..

   # 下载verl源码
   git clone https://github.com/volcengine/verl.git
   cd verl
   git checkout 54c9b7364c2d188b2ba4107404cfa3c2b446df19
   cp -r verl ../cann-recipes-train/llm_rl/deepseek/
   cd ..

   # 下载vLLM源码
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout v0.9.1
   cp -r vllm ../cann-recipes-train/llm_rl/deepseek/
   cd ..

   # 下载vLLM-Ascend源码
   git clone -b v0.9.1rc2 https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   git apply ../cann-recipes-train/llm_rl/deepseek/vllm_ascend_patches/0001-FIX-KVcache-NZ.patch
   cp -r vllm_ascend ../cann-recipes-train/llm_rl/deepseek/
   cd ..

   # 下载Megatron-LM源码
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_r0.8.0
   cp -r megatron ../cann-recipes-train/llm_rl/deepseek/
   cd ..

   # 下载MindSpeed源码
   git clone https://gitcode.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout v2.0.0_core_r0.8.0
   cp -r mindspeed ../cann-recipes-train/llm_rl/deepseek/
   cd ..

   # 下载MindSpeed-LLM源码
   git clone https://gitcode.com/ascend/MindSpeed-LLM.git
   cd MindSpeed-LLM
   git checkout v2.0.0
   cp -r mindspeed_llm ../cann-recipes-train/llm_rl/deepseek/

   # 回到项目目录
   cd ../cann-recipes-train/llm_rl/deepseek/
   ```

5. 修改verl代码。

   为了使能patch修改，需要修改以下verl源码。
   ```python
   # 在verl/workers/megatron_workers.py: line21处插入以下代码
   from verl_patches import prelude_patch
   ```
