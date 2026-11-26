# syntax=docker/dockerfile:1
ARG BASE_IMAGE=quay.io/ascend/vllm-ascend
ARG BASE_IMAGE_TAG=v0.30.0rc2-a5
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}
LABEL org.opencontainers.image.licenses="Apache-2.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

ARG PIP_INDEX_URL="https://mirrors.huaweicloud.com/repository/pypi/simple"
ARG ARCH="x86_64"
ARG APTMIRROR=""
ARG YUMMIRROR="https://mirrors.huaweicloud.com/openeuler"

ARG VLLM_REPO="https://gitcode.com/vllm-project/vllm.git"
ARG VLLM_ASCEND_REPO="https://gitcode.com/gh_mirrors/vl/vllm-ascend.git"
ARG TORCHTITAN_REPO="https://gitcode.com/GitHub_Trending/to/torchtitan.git"
ARG TORCHTITAN_NPU_REPO="https://gitcode.com/cann/torchtitan-npu.git"
ARG VLLM_COMMIT=v0.30.0
ARG VLLM_ASCEND_COMMIT=7e37e29a7803d6897a7113088e2b771f83205bc0
ARG TORCHTITAN_COMMIT=086bf6c166ec85c1298eb5596fa9bf95f6a2d840
ARG TORCHTITAN_NPU_COMMIT=37a9d34b9752366c2245a988ef603bce8b8753a8

ARG VIME_FORGE_REPO="https://gitcode.com/cann/cann-recipes-train"
ARG VIME_FORGE_COMMIT=master

ARG TORCH_2_14=torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_${ARCH}.whl
ARG TORCH_NPU_2_14=torch_npu-2.14.0.*-cp312-cp312-manylinux_2_28_${ARCH}.whl
ARG TORCH_2_13=torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_${ARCH}.whl
ARG TORCH_NPU_2_13=torch_npu-2.13.0.*-cp312-cp312-manylinux_2_28_${ARCH}.whl
# https://github.com/flashserve/flash-linear-attention-npu/releases/download/v26.9.1-20261013-142848/flash_linear_attention_npu_a5-26.9.1+dev36e67d2-py3-none-manylinux_2_34_${ARCH}.whl
ARG FLA_NPU=flash_linear_attention_npu_a5-*-py3-none-manylinux_2_34_${ARCH}.whl

ENV DEBIAN_FRONTEND=noninteractive \
    SOC_VERSION=ascend950dt_9582 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_TLS_VERIFY=0
# ENV HTTP_PROXY="" HTTPS_PROXY="" NO_PROXY="" http_proxy="" https_proxy="" no_proxy=""

# System and pip dependencies.
RUN if [ ${ARCH} = "aarch64" ];then \
        echo "sslverify=false" >>/etc/yum.conf && \
        if [ -n "$YUMMIRROR" ];then \
            sed -i 's#https\?://repo\.openeuler\.org#$YUMMIRROR#g' /etc/yum.repos.d/*.repo; \
        fi && \
        yum update -y && \
        yum install -y && \
        cmake curl git ninja-build patch rsync wget bash-completion htop && \
        printf '\nif ! shopt -oq posix; then\n  if [ -f /usr/share/bash-completion/bash_completion ]; then\n    . /usr/share/bash-completion/bash_completion\n  fi\nfi\n' >> /etc/bash.bashrc; \
    else \
        if [ -n "$APTMIRROR" ];then \
            sed -i "s@^\(deb.*\)https\?://[a-z0-9.-]*\.ubuntu\.com@\1$APTMIRROR@g" /etc/apt/sources.list; \
        else \
            sed -i '/ports\.ubuntu\.com/ {h;s|ports\.ubuntu\.com|mirrors.tuna.tsinghua.edu.cn|g;G}' /etc/apt/sources.list; \
        fi && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            build-essential cmake curl git libnuma-dev ninja-build patch rsync wget bash-completion htop && \
        printf '\nif ! shopt -oq posix; then\n  if [ -f /usr/share/bash-completion/bash_completion ]; then\n    . /usr/share/bash-completion/bash_completion\n  fi\nfi\n' >> /etc/bash.bashrc && \
        rm -rf /var/lib/apt/lists/*; \
    fi
RUN pip config set global.index-url ${PIP_INDEX_URL} && \
    if [ "${PIP_INDEX_URL#http://}" != "${PIP_INDEX_URL}" ]; then pip config set global.trusted-host "$(echo ${PIP_INDEX_URL} | awk -F/ '{print $3}')"; fi && \
    pip config set global.extra-index-url \
    "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi" && \
    git config --global http.version HTTP/1.1 && git config --global http.sslVerify false && git config --global https.sslVerify false


# The token is mounted as a BuildKit secret for this RUN only, so it never
# lands in a layer or in the image history:
#   docker build --secret id=gitcode_token,src=/path/to/token ...
RUN --mount=type=secret,id=gitcode_token,required=false \
    export GITCODE_API_TOKEN="$(cat /run/secrets/gitcode_token 2>/dev/null || true)" && \
    export GIT_TERMINAL_PROMPT=0 && \
    rm -rf /app/* && \
    git -c credential.helper='!f(){ echo username=oauth2; echo "password=$GITCODE_API_TOKEN"; }; f' clone ${VLLM_REPO} "vllm" && \
    git -c credential.helper='!f(){ echo username=oauth2; echo "password=$GITCODE_API_TOKEN"; }; f' clone ${VLLM_ASCEND_REPO} "vllm-ascend" && \
    git clone ${TORCHTITAN_REPO} "torchtitan" && \
    git -c credential.helper='!f(){ echo username=oauth2; echo "password=$GITCODE_API_TOKEN"; }; f' clone ${TORCHTITAN_NPU_REPO} "torchtitan-npu" && \
    git -C "vllm" checkout ${VLLM_COMMIT} && git -C "vllm-ascend" checkout ${VLLM_ASCEND_COMMIT} && \
    git -C "torchtitan" checkout ${TORCHTITAN_COMMIT} && git -C "torchtitan-npu" checkout ${TORCHTITAN_NPU_COMMIT} && \
    git clone ${VIME_FORGE_REPO} "cann-recipes-train" && \
    git -C "cann-recipes-train" checkout ${VIME_FORGE_COMMIT}

COPY npu_patch/ /app/npu_patch/
RUN pip install --requirement /app/npu_patch/requirements.txt --no-deps --no-build-isolation -i ${PIP_INDEX_URL} && \
    pip install /app/npu_patch/${TORCH_2_13} /app/npu_patch/${TORCH_NPU_2_13} triton_ascend==3.6.0 /app/npu_patch/${FLA_NPU} && \
    python3 -c 'import torch, triton, fla_npu;'

# Vime currently imports torch_memory_saver from its training actor.
RUN git clone https://github.com/sgl-project/sgl-kernel-npu.git "sgl-kernel-npu" && \
    git -C "sgl-kernel-npu" checkout 2026.9.0.post10 && \
    cd sgl-kernel-npu &&  bash build.sh -a memory-saver && \
    pip install --no-deps output/torch_memory_saver-*.whl && \
    cd /app && rm -rf sgl-kernel-npu

# vllm and vllm-ascend are installed editable (-e) in the base image;
RUN cd /app && pip uninstall vllm vllm-ascend -y && \
source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && \
    git -C vllm apply --whitespace=nowarn /app/npu_patch/vllm.patch && \
    git -C vllm-ascend apply --whitespace=nowarn /app/npu_patch/vllm-ascend.patch && \
    git -C torchtitan-npu apply --whitespace=nowarn /app/npu_patch/torchtitan-npu.patch && \
    VLLM_TARGET_DEVICE=empty pip install --no-deps --no-build-isolation -e vllm && \
    pip install --no-deps --no-build-isolation -e vllm-ascend && \
    pip install --no-deps --no-build-isolation -e torchtitan && \
    pip install --no-deps --no-build-isolation -e torchtitan-npu && \
    pip install --no-deps --no-build-isolation -e /app/torchtitan-npu/experiments/torchao-npu && \
    rm -rf /app/vllm/rust/target /app/vllm-ascend/csrc/build /app/vllm-ascend/build_out && \
    python3 -c 'import torch, torch_memory_saver, vllm, vllm_ascend;'


# ---- /app/uv-train: training-side uv venv (torch 2.14) ----------------------
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    uv venv /app/uv-train --python "$(command -v python3)" --system-site-packages && \
    uv pip install -i ${PIP_INDEX_URL} --python /app/uv-train/bin/python setuptools-rust && \
    uv pip install --python /app/uv-train/bin/python --torch-backend cpu --no-deps /app/npu_patch/${TORCH_2_14} /app/npu_patch/${TORCH_NPU_2_14} && \
    /app/uv-train/bin/python -c 'import torch, torch_npu, torchao_npu; print("uv-train:", torch.__version__, torch_npu.__version__)' && \
    uv venv /app/uv-vllm --python "$(command -v python3)" --system-site-packages && \
    uv pip install -i ${PIP_INDEX_URL} --python /app/uv-vllm/bin/python setuptools-rust && \
    rm -rf /app/npu_patch /root/.cache /root/ascend /tmp/*

ENTRYPOINT []
CMD ["/bin/bash"]