set -euo pipefail

# 可以用这个变量自定义下载路径。
repo=./torchtitan-npu

# 固定 torchtitan-npu 版本，保证环境可复现
TORCHTITAN_NPU_REV=60f5a07

if [ -d "$repo/.git" ]; then
    echo "Using existing repository: $repo"
else
    git clone https://gitcode.com/cann/torchtitan-npu.git "$repo"
fi

cd "$repo"

git checkout "$TORCHTITAN_NPU_REV"

echo "Repo: $repo"
echo "Commit: $(git rev-parse HEAD)"

source /home/developer/Ascend/cann-9.0.0/set_env.sh

pip install -r requirements.txt
pip install -e .

pip list | grep -E 'torch|torch_npu|torchtitan|triton-ascend|safetensors'

pip install --user --force-reinstall --no-deps pyarrow==21.0.0

pip install -U uv openai mcp httpx tenacity textarena==0.7.4
