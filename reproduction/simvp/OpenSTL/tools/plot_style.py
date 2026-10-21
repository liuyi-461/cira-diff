#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""matplotlib 绘图统一样式（**中文字体** + 锐度定义）。

为什么需要这个模块
------------------
matplotlib 默认字体是 DejaVu Sans，**不含 CJK 字形**。本项目的绘图脚本多处使用
中文标题/图例，若不显式设置字体，生成图上所有中文都会渲染成**方框（□）**，
且 matplotlib 只在 stderr 打印 "Glyph ... missing from font(s)" 告警——
如果日志不作数就完全看不出来，图会静默劣化。

用法（在导入 pyplot 之后、绘图之前调用一次）::

    from plot_style import use_cjk_font
    use_cjk_font()

本机已确认可用字体系列：``Noto Sans CJK SC`` / ``Noto Sans CJK JP`` /
``Droid Sans Fallback``（见 ``fc-list :lang=zh``）。
"""

import numpy as np

# 优先级：先中文，再回退到通用无衬线（即使中文缺失也不会崩，只是方框）
_CJK_CANDIDATES = [
    'Noto Sans CJK SC',      # 简体中文（首选）
    'Noto Sans CJK JP',
    'Noto Sans CJK TC',
    'Droid Sans Fallback',
    'WenQuanYi Zen Hei',
    'SimHei',
    'Microsoft YaHei',
    'DejaVu Sans',           # 最后回退
]

FONT_CALLED = False


def use_cjk_font(verbose: bool = False):
    """设置 matplotlib 使用支持中文的字体，并修正负号显示。"""
    global FONT_CALLED
    import matplotlib
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [n for n in _CJK_CANDIDATES if n in available]
    if not chosen:                       # 一个都没有时保持默认，避免设成不存在的字体
        chosen = ['DejaVu Sans']
        if verbose:
            print('[plot_style][WARN] 未找到任何中文字体，图上中文可能显示为方框')
    matplotlib.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
    # 负号：中文字体常缺 U+2212，用 ASCII 连字符替代可避免方框
    matplotlib.rcParams['axes.unicode_minus'] = False
    FONT_CALLED = True
    if verbose:
        print(f'[plot_style] 使用字体：{chosen[:2]}')
    return chosen


def sharpness(img):
    """(Laplacian 方差, 平均梯度幅值)；两者都是**越大越锐**。

    与 `tools/compare_arms.py` 的锐度比口径互补：
      * 本函数给绝对锐度（单位随图像而定）
      * compare_arms 给**相对真实图**的比值（无量纲，更利于跨样本比较）
    """
    from scipy.ndimage import laplace
    f = np.asarray(img, dtype=np.float64)
    lap = laplace(f)
    gy, gx = np.gradient(f)
    return float(lap.var()), float(np.sqrt(gx ** 2 + gy ** 2).mean())
