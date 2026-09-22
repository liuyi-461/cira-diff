# Temporal Metrics

候选指标包括 lead-time error curves、frame-to-frame tendency error、temporal correlation、object track consistency 和 rollout drift diagnostics。

状态：`PLANNED`。

Protocol 必须区分 teacher-forced one-step evaluation 与 free-running autoregressive rollout，并报告实际 lead-time index，而不是只报告最终 3 小时 aggregate。
