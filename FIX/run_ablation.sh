#!/bin/bash

# 确保脚本抛出错误时停止
set -e

echo "开始运行全套消融实验 (Total: 8 Experiments)..."
echo "代码目录: FIX/"
echo "日志目录: experiments/Ablation_Full/"
echo "========================================================"

# --------------------------------------------------------
# 0. Baseline (基准模型)
# 配置: RevIN + PageRank + ST-Attn + PatchDecoder + Gate
# --------------------------------------------------------
echo "[1/8] Running Exp 0: Baseline (The Ultimate Model)..."
python FIX/run.py --exp 0_baseline

# --------------------------------------------------------
# 1. 验证 RevIN 的有效性
# 配置: 去除 RevIN (No Norm)
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[2/8] Running Exp 1: Ablation - No RevIN..."
python FIX/run.py --exp 1_no_revin

# --------------------------------------------------------
# 2. 验证 PageRank Encoder 的有效性
# 配置: 替换为 DCRNN (有限步扩散)
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[3/8] Running Exp 2: Ablation - Replace with DCRNN..."
python FIX/run.py --exp 2_dcrnn

# --------------------------------------------------------
# 3. 验证 ST-Attention (Pre-Encoder) 的有效性
# 配置: 去除 ST-Transformer Block
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[4/8] Running Exp 3: Ablation - No ST-Attention..."
python FIX/run.py --exp 3_no_st

# --------------------------------------------------------
# 4. 验证 Patch Decoder 的有效性
# 配置: 替换为 Standard CausalConv (不分块)
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[5/8] Running Exp 4: Ablation - Standard CausalConv (No Patch)..."
python FIX/run.py --exp 4_causal

# --------------------------------------------------------
# 5. 验证 Decoder 架构选择
# 配置: 替换为 Transformer Decoder
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[6/8] Running Exp 5: Ablation - Transformer Decoder..."
python FIX/run.py --exp 5_transformer

# --------------------------------------------------------
# 6. 验证 门控机制 (Gating) 的有效性
# 配置: 去除 Gate (直接相加自适应图)
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[7/8] Running Exp 6: Ablation - No Gating..."
python FIX/run.py --exp 6_no_gate

# --------------------------------------------------------
# 7. 扩展实验: Beta 概率分布预测
# 配置: 输出 Alpha/Beta 参数，使用物理 Loss
# --------------------------------------------------------
echo "--------------------------------------------------------"
echo "[8/8] Running Exp 7: Experimental - Beta Distribution Head..."
python FIX/run.py --exp 7_beta

echo "========================================================"
echo " 所有 8 组实验运行完毕！"
echo "请检查 'final_ablation_results.csv' 查看汇总指标。"