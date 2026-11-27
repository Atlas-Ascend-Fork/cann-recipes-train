# 任务

## 任务概述

你的任务是编写自定义 AscendC 算子，你需要为给定的算子生成 cann-bench direct-launch 模式所需的 **kernel**、**launch_h**、**plugin**、**cmake** 4 个交付件。

AscendC 算子的开发应以功能正确性为首要目标，确保代码可成功编译且计算精度满足全部要求；在此基础上，进一步进行性能优化，力求获得更优的执行效率。


### 硬件平台说明

**目标硬件**：生成在 **Atlas A3（SoC 家族 `ascend910_93`）** NPU 上编译并评测的高性能算子。请针对该平台编写代码：
- 只使用在 Atlas A3 / `ascend910_93` 上有效的 AscendC API 与 intrinsic。
- 设计 tiling 与 buffer（AI Core 数量、UB / L0 / L1 容量、支持的 dtype）时请按 **Atlas A3 语义**来假定。


## 本任务算子规格等信息详述

### 算子 desc

````markdown
# Pows 算子 API 描述

## 1. 算子简介

对张量 x 的每个元素，以标量 exponent 为指数做幂运算。

**主要应用场景**：

- 张量按统一标量指数求幂（平方、立方、开方、倒数等）
- 需要将一个标量指数广播到整张量的逐元素幂运算

**算子特征**：

- 难度：L1
- x 为数据张量，exponent 为标量属性，其值广播为 x 的指数

## 2. 算子定义

### 数学公式

$$
y = x^{\;exponent}, \quad \text{即} \quad y_i = x_i^{\;exponent}
$$

其中 x 为底数张量，exponent 为标量指数属性，该指数广播到 x 的每个元素求幂；求幂在 float32 下完成后转回 x 的 dtype，输出 y 的 dtype 与 x 一致。

## 3. 接口规范

### 算子原型

```python
cann_bench.pows(Tensor x, float exponent) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| x | Tensor | 底数张量 |
| exponent | float | 标量指数，广播到 x 每个元素 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与 x 一致 | 与 x 一致 | 求幂结果 |

### 数据类型

| 输入 x dtype | 输出 y dtype |
|-----------------|-------------|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- x、y 的 dtype 一致
- exponent 为标量属性，取其值作为统一指数
- 输出 shape 与 x 一致
- **exponent 只支持 6 个离散取值**：`-2.0`、`-1.0`、`-0.5`、`0.5`、`2.0`、`3.0`，
  分别对应 `1/x²`、`1/x`、`1/√x`、`√x`、`x²`、`x³` 六条专用实现路径。
  这 6 条是算子的**全部**计算路径，没有通用指数兜底路径；通用指数 `x^p`
  在真实框架里由上层其它算子承担，不在本算子内部。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（x） | 1 ~ 8 | 逐元素计算 |
| `exponent` | `-2.0` / `-1.0` / `-0.5` / `0.5` / `2.0` / `3.0` | 六条专用路径，无通用指数路径 |

## 4. 精度要求

采用生态算子精度标准进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$\text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})$$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$\text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})$$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---|---|---|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.rand(1024, 1024, dtype=torch.float16, device="npu") + 0.5
y = cann_bench.pows(x, 2.0)  # 每个元素求平方
```

````

### 标准 golden 代码

```python
import torch

"""
Pows 算子 Torch Golden 参考实现

对输入张量 x 的每个元素，以标量 exponent 为指数做幂运算。
公式: y_i = x_i ^ exponent
"""
def pows(x: torch.Tensor, exponent: float) -> torch.Tensor:
    """
    逐元素求幂，底数为张量 x，指数为标量 exponent。

    公式: y_i = x_i ^ exponent

    Args:
        x: 底数张量
        exponent: 标量指数

    Returns:
        y: 求幂结果，dtype 与 x 一致
    """
    out_dtype = x.dtype
    y = torch.pow(x.to(torch.float32), float(exponent))
    return y.to(out_dtype)

```

### 算子 proto

```yaml
operator:
  name: Pows
  category: Elementwise
  difficulty: L1
  formula: "y = x ^ exponent, 即 y_i = x_i ^ exponent（内部按 float32 求幂后转回 x 的 dtype）"
  description: 对张量 x 的每个元素，以标量 exponent 为指数求幂
  shape_support: x 为任意形状张量；exponent 为标量属性，广播为 x 的指数
  attrs:
  - name: exponent
    type: float
    description: >-
      标量指数，广播到 x 每个元素。算子契约只支持 6 个离散取值：-2.0 / -1.0 / -0.5 / 0.5 / 2.0 / 3.0
      （依据 op_kernel/pows_base.h 的 ComputePowsBase，该函数是一条 if / else-if 链，
      只对这 6 个取值各有一个分支，且没有 else 分支；其余指数不写输出缓冲区）。
      通用指数 x^p 在真实框架里由上层其它算子承担，不在 Pows 内部。
    required: true
  inputs:
  - name: x
    description: 底数张量
    dtype:
    - float16
    - bfloat16
    - float32
  outputs:
  - name: y
    description: 求幂结果，dtype 与 x 一致
    dtype:
    - float16
    - bfloat16
    - float32
  schema: pows(Tensor x, float exponent) -> Tensor y

```

### 算子 cases

```yaml
cases:
- operator: Pows
  case_id: 1
  input_shape:
  - [1048576]
  dtype:
  - float16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 3]
  note: "S-float16-1M-对齐-1D-平方"
- operator: Pows
  case_id: 2
  input_shape:
  - [2048, 2048]
  dtype:
  - float32
  attrs: {exponent: 3.0}
  value_range:
  - [0.5, 3]
  note: "M-float32-4M-对齐-2D-立方"
- operator: Pows
  case_id: 3
  input_shape:
  - [4096, 4096]
  dtype:
  - bfloat16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "M-bfloat16-16M-对齐-2D"
- operator: Pows
  case_id: 4
  input_shape:
  - [1023, 1023]
  dtype:
  - float16
  attrs: {exponent: 0.5}
  value_range:
  - [1, 2]
  note: "S-float16-1M-非对齐-2D-sqrt"
- operator: Pows
  case_id: 5
  input_shape:
  - [1009, 1021]
  dtype:
  - float32
  attrs: {exponent: -1.0}
  value_range:
  - [0.5, 2]
  note: "S-float32-1M-质数-2D-倒数"
- operator: Pows
  case_id: 6
  input_shape:
  - [8, 8, 64, 64]
  dtype:
  - float32
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "M-float32-2M-4D"
- operator: Pows
  case_id: 7
  input_shape:
  - [363, 367, 373]
  dtype:
  - bfloat16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "L-bfloat16-50M-质数-3D"
- operator: Pows
  case_id: 8
  input_shape:
  - [2, 7, 256, 256]
  dtype:
  - float16
  attrs: {exponent: 3.0}
  value_range:
  - [0.5, 2]
  note: "S-float16-917K-4D-立方"
- operator: Pows
  case_id: 9
  input_shape:
  - [11, 13, 17, 67]
  dtype:
  - float32
  attrs: {exponent: -0.5}
  value_range:
  - [0.5, 2]
  note: "M-float32-质数-4D-rsqrt"
- operator: Pows
  case_id: 10
  input_shape:
  - [1000003]
  dtype:
  - float16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "S-float16-1M-质数-1D"
- operator: Pows
  case_id: 11
  input_shape:
  - [255, 8193]
  dtype:
  - bfloat16
  attrs: {exponent: -2.0}
  value_range:
  - [0.5, 2]
  note: "S-bfloat16-2M-2D-负平方"
- operator: Pows
  case_id: 12
  input_shape:
  - [4097, 511]
  dtype:
  - float32
  attrs: {exponent: 0.5}
  value_range:
  - [0.5, 2]
  note: "S-float32-2M-sqrt-2D"
- operator: Pows
  case_id: 13
  input_shape:
  - [1024]
  dtype:
  - float16
  attrs: {exponent: 2.0}
  value_range:
  - [0, 0]
  note: "S-float16-零底数-1D"
- operator: Pows
  case_id: 14
  input_shape:
  - [1024]
  dtype:
  - float32
  attrs: {exponent: 3.0}
  value_range:
  - [1, 1]
  note: "S-float32-底数1-1D"
- operator: Pows
  case_id: 15
  input_shape:
  - [2049, 513]
  dtype:
  - float16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "S-float16-1M-非对齐-2D-b"
- operator: Pows
  case_id: 16
  input_shape:
  - [4, 5, 6, 7, 8]
  dtype:
  - float32
  attrs: {exponent: 3.0}
  value_range:
  - [0.5, 2]
  note: "S-float32-小-5D"
- operator: Pows
  case_id: 17
  input_shape:
  - [100000]
  dtype:
  - bfloat16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "M-bfloat16-100K-1D"
- operator: Pows
  case_id: 18
  input_shape:
  - [333, 333]
  dtype:
  - float32
  attrs: {exponent: 0.5}
  value_range:
  - [0.5, 2]
  note: "S-float32-质数-2D-非整数幂"
- operator: Pows
  case_id: 19
  input_shape:
  - [777]
  dtype:
  - float16
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "S-float16-小-1D"
- operator: Pows
  case_id: 20
  input_shape:
  - [3, 7, 11, 13]
  dtype:
  - float32
  attrs: {exponent: 2.0}
  value_range:
  - [0.5, 2]
  note: "S-float32-质数-4D"

```

## 输出格式

输出只需 **4 段** 代码块:

    kernel_src
    ```cpp
    ...
    ```

    launch_h_src
    ```cpp
    ...
    ```

    plugin_src
    ```cpp
    ...
    ```

    cmake_src
    ```cmake
    ...
    ```

第一行第一个非空字符必须是 `k` (来自 `kernel_src`)。


## 示例代码（仅作格式模板，不是你要生成的算子）

样例算子 `sqrt` 代码演示 4 个代码交付件的结构，**看结构、勿抄语义**。

以下是样例算子 `sqrt` 的 4 个代码交付件：

kernel_src

```cpp
#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

template <typename T>
class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
        zGm_.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
        // tileSize is element count (not bytes)
        pipe_.InitBuffer(inQueueX_,  PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, PIPELINE_DEPTH, tileSize * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(zFloatBuf_, tileSize * sizeof(float));
        }
        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        if (currentBlockLength < 0) currentBlockLength = 0;
        elementNumPerTile_ = tileSize;
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = inQueueX_.DeQue<T>();
        auto zLocal = outQueueZ_.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Sqrt(zLocal, xLocal, count);
        } else {
            auto xF = xFloatBuf_.Get<float>();
            auto zF = zFloatBuf_.Get<float>();
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Sqrt(zF, xF, count);
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(zLocal, zF, roundMode, count);
        }
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto zLocal = outQueueZ_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, zGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, zFloatBuf_;
    int64_t elementNumPerTile_ = 0, tileNum_ = 0, tailTileElementNum_ = 0;
};

template <typename T>
__global__ __aicore__ __vector__ void sqrt_kernel(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    KernelSqrt<T> op;
    op.Init(x, z, totalLength, blockLength, tileSize);
    op.Process();
}

// Returns (numBlocks, blockLength, tileSize) where tileSize is element count
// per UB tile. 2048 comfortably fits the two in/out queues plus the two fp32
// cast buffers on 910b's UB.
std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static uint32_t FIXED_TILE_ELEMS = 2048;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, blockLength, static_cast<int64_t>(FIXED_TILE_ELEMS));
}

extern "C" {

void launch_sqrt_kernel_float(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

void launch_sqrt_kernel_half(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

void launch_sqrt_kernel_bfloat16(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    sqrt_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize);
}

}

```

launch_h_src

```cpp
#ifndef SQRT_LAUNCH_H
#define SQRT_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_sqrt_tiling_params(int64_t totalLength);

extern "C" {
void launch_sqrt_kernel_float   (GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_sqrt_kernel_half    (GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_sqrt_kernel_bfloat16(GM_ADDR x, GM_ADDR z, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

#endif // SQRT_LAUNCH_H

```

plugin_src

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/sqrt_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("sqrt(Tensor x) -> Tensor");
}

torch::Tensor sqrt_meta(const torch::Tensor &x)
{
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("sqrt", sqrt_meta);
}

torch::Tensor sqrt_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = sqrt_meta(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_sqrt_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        auto dtype = x.scalar_type();
        if      (dtype == torch::kFloat32) launch_sqrt_kernel_float   (x_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        else if (dtype == torch::kFloat16) launch_sqrt_kernel_half    (x_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        else if (dtype == torch::kBFloat16) launch_sqrt_kernel_bfloat16(x_ptr, z_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Sqrt", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("sqrt", sqrt_npu);
}

} // namespace cann_bench

```

cmake_src

```cmake
set(SQRT_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/sqrt_kernel.cpp
)

set(SQRT_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/sqrt_plugin.cpp
)

register_direct_launch_op(
    "${SQRT_KERNEL_SRCS}"
    op_kernel
    "${SQRT_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)

```
