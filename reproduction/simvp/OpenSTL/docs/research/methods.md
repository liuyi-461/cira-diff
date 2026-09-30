# 方法设计：SimVP

## 方法原理
SimVP 将视频预测分解为：空间编码（下采样提取隐表示）→ 隐空间时序/空间翻译
（MetaFormer 骨干堆叠）→ 空间解码（上采样 + skip 连接还原分辨率）。

## 输入输出
- 输入：`x_raw` 形状 `(B, T, C, H, W)`，T = pre_seq_length。
- 输出：`(B, T_out, C, H, W)`，T_out = aft_seq_length。

## 算法流程
1. 将时间维拼入 batch：`x = x_raw.view(B*T, C, H, W)`。
2. `Encoder`：`N_S` 层 `ConvSC`（交替 2× 下采样），输出 latent 与 skip(`enc1`)。
3. `MidMetaNet`：将 `(B, T, C_, H_, W_)` reshape 为 `(B, T*C_, H_, W_)`，
   经 `N_T` 个 `MetaBlock`（默认 gSTA）翻译。
4. `Decoder`：`N_S` 层 `ConvSC`（上采样），末层 `hid + enc1` 残差，1×1 卷积读头输出。
5. reshape 回 `(B, T, C, H, W)`。

## 参数设置（默认 gSTA，mmnist）
`hid_S=64, hid_T=512, N_S=4, N_T=8, spatio_kernel_enc/dec=3, lr=1e-3,
batch_size=16, drop_path=0, sched='onecycle'`。

> 当 `aft_seq_length > pre_seq_length` 时，`SimVP.forward` 采用递归自回归拼接，
> 以保证与自回归方法的可比性（DEC-002）。
