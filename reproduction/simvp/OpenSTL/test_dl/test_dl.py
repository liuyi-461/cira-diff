#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小深度学习训练测试脚本
验证：CUDA 可用性 + GPU 识别 + 前向/反向传播 + 多轮训练是否能跑通
"""

import os
import sys
import time

import torch
import torch.nn as nn


def main():
    print("=" * 60)
    print("[ENV] python     :", sys.version.split()[0])
    print("[ENV] torch      :", torch.__version__)
    print("[ENV] cuda avail :", torch.cuda.is_available())
    print("[ENV] cuda ver   :", torch.version.cuda)
    print("[ENV] device cnt :", torch.cuda.device_count())

    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用，请检查驱动 / 环境")
        sys.exit(1)

    for i in range(torch.cuda.device_count()):
        print(f"[ENV] GPU {i}      : {torch.cuda.get_device_name(i)}")

    device = torch.device("cuda:0")
    print(f"[ENV] using      : {device}")
    print("=" * 60)

    # 一个简单的 MLP
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 64
    x = torch.randn(batch_size, 1024, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    epochs = 20
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        mem = torch.cuda.memory_allocated(device) / 1024**2
        print(f"[TRAIN] epoch {epoch+1:02d}/{epochs}  loss={loss.item():.4f}  "
              f"allocated={mem:.1f} MB")

    torch.cuda.synchronize()
    t1 = time.time()
    print("=" * 60)
    print(f"[DONE] 训练 {epochs} 轮完成，用时 {t1 - t0:.2f}s")
    print(f"[DONE] 峰值显存 {torch.cuda.max_memory_allocated(device)/1024**2:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
