# Reproduction 文档

可执行记录和 provenance 位于顶层 [`reproduction/`](../../reproduction/) 目录。本页规定阅读顺序：

1. 验证论文和官方 repository；
2. 锁定 upstream revision 和 license；
3. 记录 dataset/task/input/output 语义；
4. 验证 environment 和 checkpoint availability；
5. 在 local adaptation 前先运行 reference inference；
6. 记录 adapter、patch、test 和 artifact；
7. 只有 verified reusable behavior 才能 promotion。

状态必须区分 `Surveyed`、`Code Available`、`Downloaded`、`Environment Working`、`Inference Reproduced`、`Training Reproduced` 和 `Evaluation Reproduced`。
