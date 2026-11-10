# CANN-Bench 评测与评分

本模块用于把一份 AscendC 算子实现提交给已有的 CANN-Bench 服务，等待编译、精度和性能结果，并按需计算训练 reward。可以独立使用，不需要启动 Vime 训练、Agent 或 Ray。

评测与评分分开进行：评测返回服务的实际结果，评分函数将有效结果转换为 reward。同一份结果可以离线比较不同评分策略，不必重新编译或运行算子。完整的单轮训练流程见[算子生成强化学习样例](../cann_bench_single_turn/README.md)。

## 在训练样例中使用

在启动命令中设置 `CANN_EVAL_URL` 指向已有的 CANN-Bench 服务，其他评测选项见[任务配置](../cann_bench_single_turn/config.yaml)。服务需支持 `/api/v1` 接口和 `1.x` 结果格式；本模块不负责部署服务。

单轮样例会自动将模型生成的四文件代码转换为提交格式，提交评测、等待结果并归档。运行样例时无需手动生成 TXT、调用 Python API 或另写结果保存代码。完整步骤见[训练指南](../cann_bench_single_turn/README.md)。

每个评测任务通过服务返回的 `job_id` 关联结果；实验 alias 仅用于区分运行，不改变题目算子名。

## 阅读评测结果

| 字段 | 如何理解 |
| --- | --- |
| `job_id` | 服务受理的任务标识，便于查找报告和日志 |
| `verdict` | 评测结论，例如 `passed`、`compile_fail`、`accuracy_fail` |
| `compiled` | 是否编译成功；`None` 表示没有得到该信息 |
| `case_pass / case_total` | 精度通过的 case 数与总数 |
| `score` | 服务原始分数，不等于训练 reward；`None` 表示未获得分数 |
| `speedup`、`case_details` | 汇总加速比及逐 case 结果；缺测不能当作加速比 0 |
| `infrastructure_failure`、`retryable` | 是否存在基础设施故障或需由调用方处理的重试状态 |
| `timing` | 排队、执行和查询等阶段的耗时记录 |

先判断故障标记，再判断模型结果。格式、编译、运行或精度失败属于模型评测结果；网络、服务或环境故障不能直接解释为模型得了 0 分。即使精度达到 `20/20`，性能采集仍可能失败，应查看逐 case 信息，不能仅凭精度判断已有性能数据。

单轮样例将评测结果保存在输出目录的 `rollout_jsonl/` 中，对应记录的 `metadata.evaluation` 包含标准结果、逐 case 数据及额外诊断信息。

`EvaluationResult.envelope` 保留原始响应；`to_dict()` 将标准字段保存一份，其余原始信息放在 `diagnostics`，避免重复保存逐 case 数据。评分只读取稳定字段，不从 `raw` 猜测缺失值。

## 独立评分

评分只读取已有评测结果，不会重新提交任务、修改评测结果或访问训练状态。模型编译、精度等失败结果仍可评分；基础设施或 retryable 结果不能作为模型结果评分。Reward 不限制在 0～1，不能当作精度百分比。

### 两种评分方式

仓库提供两套可选的 reward 定义，公式与奖励目标不同：

| 项目 | `scoring.py` | `scoring_legacy.py` |
| --- | --- | --- |
| 基础分 | 服务分数除以 100 | 有服务分数时同样除以 100；缺失时按评测结论与精度通过比例回退 |
| 全精度奖励 | 额外加 `0.2` | 不额外添加 |
| 额外性能奖励 | 仅全对时考虑；服务已有 HAP 时不重复叠加 | 部分通过也可考虑；按下述数值条件判断是否需要补性能 |
| 性能数据的使用 | 优先使用有效耗时锚点公式，否则使用 Box-Cox 转换 | 优先使用逐 case 的 HAP，其次耗时锚点公式，最后用正加速比 `s / (1 + s)` |
| 缺失性能 | 不伪造测量值 | 不伪造测量值；失败或缺测 case 仍计入总 case 数 |

HAP 是服务提供的逐 case 性能评分。耗时锚点公式使用候选实现耗时、参考耗时和硬件下界；不能把不同参照下的加速比混用。

`scoring_legacy.py` 根据基础分判断是否补充性能奖励：仅当基础分与 `0.2 + 0.3 × passed / total` 的差不超过 `1e-4`，且有可计分的性能时，才补性能分，避免重复计算。性能贡献按 `0.5 / total` 加权；只有完全没有可计分的逐 case 性能时，才回退到平均加速比。

在使用加速比回退时，两者对 `0 < s < 1` 的处理不同：`scoring_legacy.py` 的 `s / (1 + s)` 仍为正；`scoring.py` 的 Box-Cox 分支为负。具体公式及边界条件见 [scoring.py](scoring.py) 和 [scoring_legacy.py](scoring_legacy.py)。

### 切换评分策略

先复制[单轮任务配置](../cann_bench_single_turn/config.yaml)，再修改副本中的以下字段；这不是完整配置文件，其他评测地址、超时和 alias 等字段应保留：

```yaml
cann_scoring:
  function_path: examples.cann_bench_evaluator.scoring_legacy.score_result
cann_single_turn:
  thinking_close_reward: false
```

从模型入口选择自己的任务配置：

```bash
bash examples/cann_bench_single_turn/run_qwen35_4b_single_node_npu_rl.sh \
  -- --custom-config-path /absolute/path/to/task.yaml
```

### `scoring_legacy.py` 公式细则

- 有服务分数时，以 `score / 100` 为基础。仅当它与
  `0.2 + 0.3 × passed / total` 相差不超过 `1e-4`，且存在可计分的性能时，
  才补性能分；否则保留服务分，避免重复计入性能。这里采用数值判断，
  **不是**用“是否有 HAP”替代判断。
- 每个通过的 case 优先使用 `hap`，其次使用正的 `candidate_us / baseline_us /
  hardware_bound_us` 重建 Eq.3：`(baseline - T_HW) / (candidate + baseline - 2 × T_HW)`；
  锚点不可用时，正加速比 `s` 使用 `s / (1 + s)`。性能贡献为这些 case 分数之和
  乘 `0.5 / total`。部分通过也可补性能，失败和未测 case 按总 case 数摊薄；
  `0 < s < 1` 仍有正贡献，与 Box-Cox 不同。
- 仅在没有任何逐 case 性能可计分时，才使用服务的成功 case 平均加速比，
  并按通过比例加权；已有部分逐 case 数据时不拿整体均值补齐缺测。
- 不增加全对 `+0.2`，不做 Box-Cox，不启用 custom 分层奖励或 stage bonus。
  缺失分数时，按 verdict/case 计算基础分回退；
  `anti_hack_fail / eval_aborted / parse_fail` 为硬零分。
- 基础设施或 retryable 结果拒绝评分，仍由任务层处理。
  模型编译/运行的诊断文本不被当作网络错误，服务原始分数、case 和日志不会被改写。

任务层的格式恢复扣分、非法输出/重复输出惩罚仍独立保留。thinking-close 也由任务层
控制；若要对齐旧实验的最终 reward，应保持上例的 `thinking_close_reward: false`，
并确认没有环境覆盖 `VIME_CANN_THINKING_CLOSE_REWARD=1`。
归档继续记录所选 `function_path`、源文件 SHA-256、`base_reward` 和
`performance_reward`。为兼容旧适配器，分数缺失时的 case 基础分回退也归入后一分项，
不表示采到了性能。

选用 `scoring.py` 时，将路径设为 `examples.cann_bench_evaluator.scoring.score_result`。`--custom-config-path` 选择任务 YAML，不能用 Titan 的 `--config` 代替。评分策略在新运行初始化时确定，修改 YAML 不会热更新正在运行的训练。

服务返回失败结果后，还会 GET `/download/{id}/logs/compile.log`（编译失败的完整编译器日志，不取外层摘要）或
`/download/{id}/logs/evaluate.log`（编译后失败），完整文本分别保存为
`compile_log` / `evaluate_log`；失败阶段不明确时尝试获取两份日志。单轮任务会随
`metadata.evaluation` 写入现有 JSONL，不只保留服务的首条错误或末尾4000字符摘要。
通过的任务不下载日志；日志获取失败记录在 `log_fetch_errors`，不改变原判定、reward
或过滤/重提逻辑，也不跟随服务返回的任意日志URL。

服务从 plugin 注册中提取提交算子。客户端同时校验受理的 job id 和题目算子名；返回算子名不一致时记为 `parse_fail`、分数为 0，不计成功奖励，也不标为基础设施故障、不触发重采。原始算子名及结果保留在诊断中的 `reported_op`、`reported_result`。名称须精确匹配，不隐式删除 `exp_` 前缀或猜测别名；`unsupported_task` 仍是有效的失败结果。

自然结束 thinking 的奖励由任务层独立控制（`VIME_CANN_THINKING_CLOSE_REWARD`），不属于这两个评分函数；格式恢复、非法或重复输出等惩罚也由任务层处理，见[Reward 与失败样本](../cann_bench_single_turn/README.md#reward-与失败样本)。

也可以提供自定义的同步评分函数，签名为 `score_result(EvaluationResult) -> Score`，再配置其导入路径。`Score` 的 reward 和各分项必须为有限数值，分项之和必须等于 reward；错误会明确报出，不会自动裁剪或变成模型零分。

## 协议边界

- GET 和 POST 可有限重试，`request_retries=2` 默认最多请求 3 次。POST 每次重建完整上传内容；网络超时/断连、HTTP 5xx/408/429 时重试，认证和不兼容协议不重试。
- 服务没有提交去重，重发可能创建重复任务。调用方只使用成功拿到 job id 的那次结果，不叠加多次 reward；未知 job id 的任务无法定向取消，会增加服务负载和网页端提交统计。
- 提交重试耗尽后返回 `submit_failed` infrastructure failure，不直接中断训练；是否过滤/补采由调用方决定。归档保留错误及提交尝试次数。
- queue 包含编译排队和等待设备；execution 包含编译、运行及终态结果获取。两种时间分别限额，耗时基于轮询观察，为近似值。
- queue 超时先 DELETE；确认成功后，最多进行 `queue_retries` 次同一答案重提交。运行/编译超时作为模型结果，不重提交。
- 已取得 job id 后，状态或结果查询遇到断连、请求超时、HTTP 5xx/408/429，保留同一任务并退避重试，不立即删除或重提；终态已确认后只重试结果获取。重试等待计入原有 queue/execution 预算；单次 HTTP 请求仍受 `request_timeout` 约束，不新增一套总超时。
- 预算耗尽时若接口仍不可用，返回未计分的 `poll_failed` / `result_failed` infrastructure failure，尽力取消已知任务，不把网络中断记成模型执行超时，也不触发安全重提。重试恢复后在 `timing.query_failure_count` 记录外层失败次数。
- 认证错误和不兼容协议不按网络故障重试。解析失败仍返回相应 infrastructure failure，保留原始响应与错误原因；错误 job id 的结果不得用于计分。
- DELETE 默认最多尝试 6 次。返回 404 时还要通过状态接口确认不存在；不把一个缺失的 DELETE 路由当成取消成功。
- 不兼容协议、invalid 评分字段返回未计分的基础设施失败，不直接中断训练、不降级为有效模型 0 分。地址/认证/调用配置错误仍明确报错，不能靠无限补采掩盖。
- speedup 数值字符串先转换；无效、非有限或非正的 speedup 不计性能分，告警包含 op、job id、原始值及类型，归档保留 `speedup_raw`。正常 null 不报警。正常失败可没有编译/分数/用例测量数据，不因此拒收。
- DELETE 失败只记录取消结果及日志，不覆盖已有评测结论、不直接中断训练；未确认删除成功时绝不安全重提。
- 环境故障/unknown 返回 infrastructure failure；格式/编译/精度等模型结果保持可区分。

## 调整等待时间与重试

评测繁忙时，先区分排队、编译执行和网络查询，再调整相应参数，不要只增加一个笼统的总超时。以下为任务配置中 `cann_evaluator` 的默认值，时间单位均为秒：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `poll_interval` | 5 | 状态轮询间隔 |
| `queue_timeout` | 1800 | 编译排队及等待设备的预算 |
| `execution_timeout` | 900 | 编译、运行及终态结果获取的预算 |
| `request_timeout` | 30 | 单次 HTTP 请求超时 |
| `request_retries` | 2 | 请求失败后的重试次数，最多请求 3 次 |
| `delete_retries` | 5 | 取消请求的重试次数，最多请求 6 次 |
| `queue_retries` | 1 | 排队超时且确认取消成功后，允许重提交同一答案的次数 |

这些值可在[任务配置](../cann_bench_single_turn/config.yaml)的 `cann_evaluator` 中覆盖。阶段耗时基于轮询观察，为近似值；查询失败后的等待计入原有阶段预算。

已拿到 `job_id` 后，暂时断连或服务繁忙会继续查询同一任务，不立即删除、重提。若直到预算耗尽接口仍不可用，返回未计分的 `poll_failed` 或 `result_failed` 基础设施故障，不把网络中断记成模型执行超时。

提交请求的响应丢失时，服务可能已经创建任务；服务没有提交去重，重试可能产生重复任务。只使用成功获得 `job_id` 的那次结果；未知任务无法定向取消，因此不要在外层无限重提。认证、地址和调用配置错误需要修正配置，不能靠重试掩盖。

只有排队超时并确认取消成功时，才按 `queue_retries` 安全重提交；编译或运行超时不自动重提交。取消失败会记录在 `cleanup` 中，不覆盖已有结论，也不视为可以安全重提。

## 失败时查看什么

- 编译失败：查看 `compile_log`；编译后的失败：查看 `evaluate_log`。客户端会尝试下载相应的完整日志，单轮样例也会随评测结果归档。
- 精度通过但没有性能：查看 `case_details`、`error` 和评测日志，区分缺少基准、profiler 采集失败与其他原因；不把缺测填成 0。
- 网络或服务故障：查看 `error`、`attempts`、`timing` 和 `cleanup`，确认受理的任务、失败阶段及取消是否成功。
- 日志下载失败：查看 `log_fetch_errors`；下载失败不会改写已有 verdict、score 或 reward。
- 注册名不一致：先核对 `plugin_src` 和题目算子名。客户端保留服务原始判定及报告的名称，不改写分数；`unsupported_task` 也是有效的模型失败结果。

客户端只提供评测事实和有限重试；基础设施故障是否过滤、跳过题组或补采，由调用任务决定。本单轮样例的处理方式见[训练指南](../cann_bench_single_turn/README.md#reward-与失败样本)。
