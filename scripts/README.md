# Scripts

Scripts 按生命周期阶段分组。已有的 `scripts/Chase_2025/` 仍是受保护的历史实现；新 script 必须先由实验记录其行为，再考虑 promotion。

- `data/`：inspection 和 preprocessing；
- `train/`：training entry points；
- `inference/`：single-step 和 rollout；
- `evaluation/`：metric 和 artifact generation。
