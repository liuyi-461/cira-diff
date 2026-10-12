#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 EXP-006（抗模糊方案对照）的结论同步到实验索引 docs/research/experiments.md。

读取 `docs/experiments/EXP-006-anti-blur/arms_metrics.json`，
按索引既有条目格式（日期/目的/数据/方法/参数/结果/结论/下一步/复现教程）
渲染，替换 `<!-- EXP006:BEGIN/END -->` 之间内容（幂等，可重复运行）。

用法::
    python tools/sync_exp006_index.py
"""

import json
import os
import os.path as osp
import sys
from datetime import datetime

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

EXP6 = osp.join(_OPENSTL_DIR, 'docs', 'experiments', 'EXP-006-anti-blur')
INDEX = osp.join(_OPENSTL_DIR, 'docs', 'research', 'experiments.md')
BEGIN, END = '<!-- EXP006:BEGIN', '<!-- EXP006:END -->'

DESC = {
    'baseline': '单步 MSE（teacher forcing），100 epoch',
    'GAN (K=6)': '多步滚动 + 判别器对抗损失（相对权重 5%）+ 双向锐度正则',
    'multi-step (p=0)': '多步 MSE，滚 6 步，teacher forcing',
    'multi-step + SS (p=0.5)': '多步 MSE + scheduled sampling 概率 0.5',
}


def main():
    mj = osp.join(EXP6, 'arms_metrics.json')
    if not osp.exists(mj):
        print('[WARN] 无 arms_metrics.json，跳过同步')
        return
    with open(mj) as f:
        m = json.load(f)
    if not m:
        print('[WARN] arms_metrics.json 为空，跳过')
        return

    arms = list(m.keys())
    base = m.get('baseline')

    L = []

    def A(s=''):
        L.append(s)

    A('# EXP-006 抗模糊方案对照（GAN vs 多步训练 / scheduled sampling）')
    A('')
    A('日期：')
    A(datetime.now().strftime('%Y-%m-%d'))
    A('')
    A('实验目的：')
    A('长时效预测"越推越模糊"（baseline 在 LT=18 的锐度比仅 0.273）。')
    A('本实验对照两类改进方向：① GAN（判别器逼迫输出更真实，生成式，会引入纹理）；')
    A('② 多步训练 ± scheduled sampling（纯监督，减少曝光偏差，不引入纹理）。')
    A('')
    A('数据：')
    A('- 评估：test 全量 1024 条，滚动 18 步（LT=1..18），逐样本 RMSE(K) 与锐度比')
    A('- 训练（GAN 臂）：train 集 35595 条（仅 1 帧真值）')
    A('- 训练（多步臂）：validation 集 800 条（有 18 帧真值，可支撑多步监督）')
    A('')
    A('方法：')
    for a in arms:
        A(f'- **{a}**：{m[a].get("desc", DESC.get(a, ""))}')
    A('')
    A('参数：')
    A('- 各臂**热启动**于同一个 baseline `best.ckpt`，梯度步数预算均为 **8000 步**')
    A('- GAN：rollout K=6、gan_weight=0.05（相对 MSE）、grad_weight=0.05、判别器带谱归一化')
    A('- 多步：rollout K=6、lr=5e-4、20 轮 × 400 步')
    A('')
    A('结果：')
    A('')
    A('| 臂 | LT=1 RMSE(K) | LT=18 RMSE(K) | LT=18 锐度比 | 相对 baseline(LT18) |')
    A('|---|---|---|---|---|')
    for a in arms:
        r1 = m[a]['rmse'][0]
        r18 = m[a]['rmse'][-1]
        s18 = m[a]['sharp'][-1]
        if a == 'baseline' or base is None:
            rel = '—（基准）'
        else:
            d = (r18 / base['rmse'][-1] - 1) * 100
            if d < -2:
                rel = f'**改善 {abs(d):.1f}%** ✅'
            elif d > 2:
                rel = f'恶化 {d:.1f}% ❌'
            else:
                rel = '持平 —'
        A(f'| {a} | **{r1:.3f}** | **{r18:.3f}** | {s18:.3f} | {rel} |')
    A('')
    A('> 锐度比 = 预测梯度幅值 / 真实梯度幅值：<1 偏模糊，≈1 与真实相当，>1 有高频噪声。')
    A('')
    A('结论：')
    A('')
    if base:
        for a in arms:
            if a == 'baseline':
                continue
            r18 = m[a]['rmse'][-1]
            s18 = m[a]['sharp'][-1]
            d = (r18 / base['rmse'][-1] - 1) * 100
            if d > 2:
                A(f'- **{a}**：RMSE 恶化 {d:.1f}% ❌；'
                  f'锐度比 {s18:.3f}'
                  f'{"（过冲为噪声）" if s18 > 1.15 else ""}')
            elif d < -2:
                A(f'- **{a}**：RMSE **改善 {abs(d):.1f}%** ✅；锐度比 {s18:.3f}')
            else:
                A(f'- **{a}**：RMSE 持平；锐度比 {s18:.3f}'
                  f'{"（仍偏模糊）" if s18 < 0.9 else ""}')
        A('')
        A('- **根本性权衡**：180 分钟后的细节不可预测，MSE 意义下的最优输出就是平滑的'
          '后验均值；任何"补细节"的做法（GAN）必然推高 MSE（感知-失真权衡）。')

        A('')
        A('下一步：')
        A('1. 若多步方向有效 → 用更大数据量（validation 全 1024 条）重训，与 baseline 公平对比；')
        A('2. 若目标是"清晰且真实的预报"本身，应改用**生成式/概率预报**路线'
          '（EDM / 扩散，正是本复现项目的论文 CIRA-Diff 的核心主张），'
          '并配套集合与感知类评价指标，而非仅看 RMSE；')
        A('3. 结论回填 `docs/research/conclusions.md`。')
        A('')
        A('复现教程:')
        A('GAN：`SCHED` 无关，`sbatch test_dl/train_gan.slurm`（ROLLOUT=6）；')
        A('多步：`SCHED=0.0 / 0.5 sbatch test_dl/train_rollout.slurm`；')
        A('对比：`tools/compare_arms.py` → `docs/experiments/EXP-006-anti-blur/`。')
        A('')
        A('> ⚠️ 对照局限：GAN 臂与多步臂**训练数据来源不同**（train 35595 条 vs '
          'validation 800 条），本实验比较的是"同热启动起点、同 8000 步预算下'
          '哪种改进方向更有效"，非严格同数据对照。')

    body = '\n'.join(L)
    with open(INDEX) as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        print('[WARN] 索引中无 EXP006 标记，跳过')
        return
    b = text.index(BEGIN)
    b_end = text.index('-->', b) + 3
    e = text.index(END)
    e_end = text.index('-->', e) + 3
    new = (text[:b] + '<!-- EXP006:BEGIN -->\n' + body + '\n'
           + '<!-- EXP006:END -->' + text[e_end:])
    with open(INDEX, 'w') as f:
        f.write(new)
    print(f'[OK] 已同步 EXP-006 到 {INDEX}')
    print('=' * 66)
    print(body[:900])
    print('=' * 66)


if __name__ == '__main__':
    main()
