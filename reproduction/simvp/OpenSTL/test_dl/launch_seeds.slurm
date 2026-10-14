#!/bin/bash
#SBATCH --job-name=launch_seeds
#SBATCH --output=test_dl/logs/launch_seeds_%j.out
#SBATCH --error=test_dl/logs/launch_seeds_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#
# 依赖 286（全量 1024 条 SS 臂的 test 评估）完成后：
#   1) 读取 EXP-006 的 arms_metrics.json，判断"全量 SS 臂"在 LT=18 是否优于 baseline
#   2) **方向正确才**自动提交 2 个补充 seed（seed 1 / seed 7），配置与全量臂完全一致
#   3) 每个 seed 训练后自动评估；两个都完成后自动跑五臂对比
#
# 这样设计的理由：若全量重训推翻原结论，就没必要再烧 ~2 小时 GPU 跑 seed。
#
# 提交：sbatch --dependency=afterany:286 test_dl/launch_seeds.slurm

set -euo pipefail

# 可移植性：仓库根按优先级取 OPENSTL_DIR > SLURM_SUBMIT_DIR（校验含 tools/ 与 openstl/）> 内置默认值。
# ⚠️ 不能用 ${BASH_SOURCE[0]} 推导：SLURM 会把脚本复制到 /var/spool/slurmd/jobNNNN/slurm_script
#    再执行，BASH_SOURCE 指向 spool 而非原路径（实测会推导出 /var/spool/slurmd，已踩坑）。
OPENSTL_DIR="${OPENSTL_DIR:-}"
if [ -z "$OPENSTL_DIR" ]; then
    for _cand in "${SLURM_SUBMIT_DIR:-}" "$(pwd)"; do
        if [ -n "$_cand" ] && [ -d "$_cand/tools" ] && [ -d "$_cand/openstl" ]; then
            OPENSTL_DIR="$_cand"
            break
        fi
    done
fi
OPENSTL_DIR="${OPENSTL_DIR:-/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL}"

# 护栏：解析出的仓库根必须含 tools/ 与配置文件，否则立即失败。
# 目的：把"路径解析错却静默跑歪"变成启动即报错（曾在 SLURM spool 路径上踩坑）。
if [ ! -d "$OPENSTL_DIR/tools" ] || [ ! -f "$OPENSTL_DIR/configs/satcast/SimVP.py" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2
    echo "[FATAL] 缺 tools/ 或 configs/satcast/SimVP.py。" >&2
    echo "[FATAL] 请从 OpenSTL 根目录执行 sbatch test_dl/xxx.slurm，或显式 export OPENSTL_DIR=<仓库根>" >&2
    exit 3
fi
LOGDIR="$OPENSTL_DIR/test_dl/logs"
EXP6="$OPENSTL_DIR/docs/experiments/EXP-006-anti-blur/arms_metrics.json"
mkdir -p "$LOGDIR"; cd "$OPENSTL_DIR"

source /home/group1/miniconda3/etc/profile.d/conda.sh && conda activate openstl

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"

# ---------- 1) 判断方向 ----------
VERDICT=$(python - "$EXP6" <<'PY'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print('NO_DATA'); raise SystemExit
m = json.load(open(p))
FULL, BASE = 'SS p=0.5 (全量1024)', 'baseline'
if FULL not in m or BASE not in m:
    print('MISSING'); raise SystemExit
f, b = m[FULL], m[BASE]
lt18_f, lt18_b = f['rmse'][-1], b['rmse'][-1]
gain = (1 - lt18_f / lt18_b) * 100
print(f'OK gain={gain:.2f}' if lt18_f < lt18_b else f'REVERSE gain={gain:.2f}')
print(f'[CHECK] LT=18: 全量SS={lt18_f:.3f} K  baseline={lt18_b:.3f} K  '
      f'相对收益={gain:+.2f}%', file=sys.stderr)
print(f'[CHECK] LT=18 锐度: 全量SS={f["sharp"][-1]:.3f}  baseline={b["sharp"][-1]:.3f}',
      file=sys.stderr)
PY
)
echo "[INFO] 判定结果: $VERDICT"

if [[ "$VERDICT" != OK* ]]; then
    echo "[SKIP] 全量臂未优于 baseline（$VERDICT），不启动补充 seed。"
    echo "[SKIP] 原 +15.1% 判定为小样本假象或方向不稳健，请人工复核。"
    exit 0
fi

# ---------- 2) 提交两个补充 seed（配置与全量臂完全一致） ----------
COMMON_OPTS=(ROLLOUT=6 SCHED=0.5 EPOCHS=16 N_TRAIN=1024 N_VAL=512 VAL_SOURCE=train)
JOBS=()
for S in 1 7; do
    EX="satcast_simvp_rollout_ss05_full_seed${S}"
    J=$(env "${COMMON_OPTS[@]}" SEED=$S EX_NAME=$EX \
        sbatch --parsable --export=ALL,SEED=$S,EX_NAME=$EX \
        "$OPENSTL_DIR/test_dl/train_rollout.slurm")
    echo "[SUBMIT] seed $S -> job $J ($EX)"
    JOBS+=("$J")
done

# ---------- 3) 两个都训完后自动评估 + 五臂对比 ----------
DEP="afterany:${JOBS[0]}:${JOBS[1]}"
sbatch --dependency="$DEP" "$OPENSTL_DIR/test_dl/eval_seeds.slurm"
echo "[SUBMIT] 汇总评估作业（依赖 $DEP）"
echo "[DONE]"
