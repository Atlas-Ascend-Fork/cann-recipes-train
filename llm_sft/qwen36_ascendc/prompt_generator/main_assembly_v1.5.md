# 任务

## 任务概述

你的任务是编写自定义 AscendC 算子，你需要为给定的算子生成 cann-bench direct-launch 模式所需的 {{TASK_DELIVERABLES}}。

AscendC 算子的开发应以功能正确性为首要目标，确保代码可成功编译且计算精度满足全部要求；在此基础上，进一步进行性能优化，力求获得更优的执行效率。


### 硬件平台说明

**目标硬件**：生成在 **Atlas A3（SoC 家族 `ascend910_93`）** NPU 上编译并评测的高性能算子。请针对该平台编写代码：
- 只使用在 Atlas A3 / `ascend910_93` 上有效的 AscendC API 与 intrinsic。
- 设计 tiling 与 buffer（AI Core 数量、UB / L0 / L1 容量、支持的 dtype）时请按 **Atlas A3 语义**来假定。


## 本任务算子规格等信息详述

### 算子 desc

````markdown
{{DESC_MD}}
````

### 标准 golden 代码

```python
{{GOLDE_PY}}
```

### 算子 proto

```yaml
{{PROTO_YAML}}
```

### 算子 cases

```yaml
{{CASES_YAML}}
```

## 输出格式

{{OUTPUT_FORMAT}}


{{ONE_SHOT}}
