# 第二台服务器：临时 E0、parent 与低维实验清单

日期：2026-10-14。本文是 [PLAN.md](PLAN.md) 的并行探索分支。
服务器 A 继续 Gate A；服务器 B 使用独立输出目录，先研究固定核心配置下
parent 的效果。本文没有假定 Gate A 已产生赢家，也没有启动第二台服务器的任务。

## 1. 分工与结论边界

- **服务器 A**：同映射/精度的 K/M 对照、M 分数变换与温度筛选、晋级、对称 contrast。
- **服务器 B**：临时 M-E0 的 parent 对照、约束是否生效的诊断、可用的层级评估；随后考虑 64 维。
- B 的结果标记为 `provisional`，不写入 A 的 `gate_a.json` 或 `gate_b.json`。
- Gate A 确定后，若核心配置改变，在新配置上补 E0 与预先选定的候选 parent，
  重新验证差值。不得将临时配置的收益直接转移到最终配置。
- B 不运行 `advance.py` 或 `add gate_b`：前者会启动 Gate A 队列，后者依赖正式赢家。
  首轮直接导入本目录的 `second_server_trials.json`。

## 2. 固定的临时 E0

| 项目 | 首轮固定值 |
|---|---|
| 模型/分数 | `lorentz_omnimodal` / M-spectral / `distance`（−D_M） |
| 映射、contrast | `direction_radius`、`text_vs_va` |
| 嵌入维度 | 512 |
| 温度 | 0.03 |
| 曲率 | 初值 1，可学习，沿用现有范围 |
| E0 entailment / consensus 权重 | 0 / 0 |
| ITM 权重 | 0.1 |
| 初始化 | 原始 VAST 权重；每个 trial 从头加载，不从另一个 trial 续训 |
| 训练 | 1,360 次有效更新，global batch 64，首轮 seed 50 |
| 学习率 | backbone 2e-5、CLIP visual 5e-7、新双曲参数 1e-4；沿用统一 warmup/scheduler |
| 输入 | TVA；训练去掉字幕，视频 2 帧，音频 1 个片段 |
| 留出集 | 2,048 个 clips；验证视频 8 帧；与 A 使用完全相同的 clip 划分 |

父节点半径下限、曲率、ITM、初始化、训练预算和数据增强在四组间保持一致。
同样的 λ 不保证同样的有效约束强度，因此首轮是固定工作点的消融，不是各 parent
分别充分调优后的比较。B 必须训练自己的 E0，不直接用 A 的绝对成绩代替配对对照。

## 3. 优先级 P0：首日四组 512 维对照

主要比较预先固定为 **Perron vs E0**，Text/SVD 是辅助对照。

| Trial | Parent | 边 | λ_ent | 更新数 |
|---|---|---|---:|---:|
| `b512_e0_s50` | 无额外约束 | 无 | 0 | 1360 |
| `b512_perron_w0.1_s50` | Perron proxy，按锥定义域施加半径下限 | P→T/V/A | 0.1 | 1360 |
| `b512_text_w0.1_s50` | 文本 | T→V、T→A | 0.1 | 1360 |
| `b512_svd_w0.1_s50` | SVD 方向＋可学习半径 | P→T/V/A | 0.1 | 1360 |

可导入清单：[second_server_trials.json](second_server_trials.json)。每组首轮只执行
`train → holdout_video → holdout_audio`，使用独立验证集比较；清单通过 `stages`
显式覆盖默认的全长 trial 测试流程，暂不批量跑测试集。

若第二台服务器吞吐接近 A 的四卡运行，可先预算 **8–16 小时**完成首轮和验收，
实际按首个训练及验证阶段计时修正。层级工具接入和补充实验不计入这个估计。

首轮检查清单：

- [ ] 四组均完成 1,360 次有效更新，有对应 checkpoint 和三个阶段的 `DONE.json`。
- [ ] 比较验证集 Stage-1、Top-50、音频 R@1；统一记录相对本机 E0 的百分点差值。
- [ ] 检查损失/梯度有限，AMP 跳步、Perron gap、流形误差与 P4 恒等式正常。
- [ ] 查看每条训练边的锥满足率、parent 半径、半径下限激活比例。
- [ ] 若需比较未加权 entailment 损失或分项梯度，先补充诊断代码并冻结新版本。
  当前日志不能保证所有 parent 都具有这些字段；缺失字段不能当成零。
- [ ] 原始 Perron proxy 与施加锥半径下限后的 parent 分开记录；后者不能代替前者验证 P4。
- [ ] 不把几十分之一个点的单 seed 差异直接写成改善，不根据测试集选择配置。

## 4. 启动前的依赖与复现检查

Git 仓库提供代码和试验清单，不包含 VAST 权重、数据媒体、旧实验模板或层级工具。
服务器 B 需要先在其已授权路径准备这些资源，并改写模板中的实际路径：

1. `train.json`、`evaluation.json`、`audio_evaluation.json` 三个模板。
   A 的原模板来自 `lorentz_omnimodal_20260929/templates/`。
   特别核对 `txt`、`vision`、`audio`、`video_overrides` 及 override 文件内部的媒体路径。
2. VAST initializer 目录及 `pretrained_weights` 中编码器、tokenizer 等资源。
3. VAST accepted annotations、视频和音频，以及后续测试阶段所需媒体。
4. 项目虚拟环境。A 使用 PyTorch 2.8.0+cu128、NumPy 1.26.4；B 记录 Python、
   PyTorch、CUDA、GPU 和依赖版本，先确认硬件兼容，再进行对照。

固定指纹（SHA256）：

| 内容 | SHA256 |
|---|---|
| VAST `model_step_204994.pt`，5,587,017,586 bytes；来自已记录的初始化审计 | `9005d283ea0de99549efbedeb7d21c8e25bea8fa41c0f6570296b0e7a055a6c5` |
| accepted annotations 文件 | `ce83a1b91fce11aed08cdc0080f96f466a668b245ad769610622b9c05a8aa5d9` |
| 训练 84,659 个唯一 clip ID | `2add934c68df797d10ace9b6990ae62f29ad24a1375d4d61cc7ac5a6ec96be80` |
| 验证 2,048 个唯一 clip ID | `ce9afd15fb79a44c41e6d90c5f8f58d0e9a4dc353eeb794fcdaa314925c90ab4` |

ID 指纹算法：`sha256(('\n'.join(sorted(set(str(row['clip_id']) for row in rows))) + '\n').encode())`。
同 seed 但媒体缺失会改变可用池，可能产生不同划分；必须核对 ID 指纹，而不只核对样本数。
绝对路径可不同，clip 划分、初始化和科学参数必须一致。

### 环境与资源

- 使用 B 的项目虚拟环境，不向系统或共享用户环境安装包。
- `PYTHONNOUSERSITE=1`，限制 CPU 解码线程；关闭外部实验日志上传。
- 使用工作区内较短的 TMPDIR，避免 Python multiprocessing 的 Unix socket 路径过长。
- global batch 固定 64：4 ranks 每卡 16，8 ranks 每卡 8；两者均需确认实际全局负样本为 64。
  梯度累积不自动等价于同批次 64 个对比候选。
- A 的四卡历史 allocator 预算为训练 37 GiB、评估 31 GiB，另需 context/NCCL 和至少 8 GiB 余量。
  这些数值是参考，B 根据实际显存和验收测量设置；不能直接套用到小显存卡。
- 每个最终模型约 5.3 GiB，验收也会写 checkpoint。检查磁盘；当前脚本每阶段还要求
  至少 58 GiB 空闲用于写出和保留余量。不会自动删除其他实验结果。
- 当前仅保存最终模型、不保存优化器。中途失败没有可恢复的 optimizer checkpoint，
  `--retry-failed` 不是断点续训；重试前必须归档旧 attempt 与日志，避免混合步数。

### 可执行入口

以下在 B 的项目根目录执行。`PYTHON`、`SUITE_B`、`LOG_B`、`TEMPLATES`、`PRETRAIN`、
`WEIGHTS`、`ANNOTATIONS`、`VIDEOS`、`AUDIOS`、`GPUS_B`、`PORT_B`、`TRAIN_CAP_GIB`、
`EVAL_CAP_GIB` 均须事先设置为 B 的实际值，不能照搬 A 的绝对路径。

```bash
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p .tmp

CUDA_VISIBLE_DEVICES='' "$PYTHON" experiments/next_steps/check_geometry.py
CUDA_VISIBLE_DEVICES='' "$PYTHON" experiments/next_steps/check_pmrl_distributed.py

"$PYTHON" experiments/next_steps/holdout_split.py \
  --annotations "$ANNOTATIONS" --vision "$VIDEOS" --audio "$AUDIOS" \
  --out "$SUITE_B/holdout" --seed 20261013 --size 2048
# 此处核对上面的 clip ID 指纹，路径不同不影响该指纹。

"$PYTHON" experiments/next_steps/suite.py --suite "$SUITE_B" init \
  --templates "$TEMPLATES" --pretrain-dir "$PRETRAIN" --weights-dir "$WEIGHTS" \
  --holdout-dir "$SUITE_B/holdout" --logs-dir "$LOG_B" \
  --train-memory-gib "$TRAIN_CAP_GIB" --eval-memory-gib "$EVAL_CAP_GIB"

# 只写入新 suite，不覆盖已有 trials.json。
"$PYTHON" - "$SUITE_B" <<'PY'
import json, pathlib, sys
target = pathlib.Path(sys.argv[1])
assert (target / 'protocol.json').is_file()
trials = json.loads(pathlib.Path('experiments/next_steps/second_server_trials.json').read_text())
with (target / 'trials.json').open('x') as stream:
    json.dump(trials, stream, indent=2)
PY

"$PYTHON" experiments/next_steps/suite.py --suite "$SUITE_B" run \
  --gpus "$GPUS_B" --port "$PORT_B" --dry-run
```

正式执行前先建立**独立 smoke suite**：从正式 train/holdout 分别取 512/64 条，
保持二者不相交；manifest 路径指向这些子集。复制四组 trial，将 `steps=2`、
`smoke=true`，`stages=['train','holdout_video','holdout_audio','native']`，其余模型参数不变。
使用该 smoke 的 `--suite` 完成四组验收；64 个候选满足 Top-50 reranking 所需数量。
检查退出码、2 次有效更新、检查点、四阶段 `DONE.json` 和显存峰值。
验收数据不能写入正式目录或参与排名。

验收通过后执行：

```bash
"$PYTHON" experiments/next_steps/suite.py --suite "$SUITE_B" run \
  --gpus "$GPUS_B" --port "$PORT_B" --stop-on-failure
"$PYTHON" experiments/next_steps/suite.py --suite "$SUITE_B" summarize
```

长期运行需按 B 的任务管理方式保存后台进程、PID、完整命令和日志。
选定需要正式报告的配置后，在已有 trial 的 `stages` 后添加 `video`、`audio`、`native`，
再运行同一 suite；已完成的训练/验证阶段会被跳过。测试结果不参与后续调参。

## 5. P1：根据首轮结果做有界补充

| 情况 | 下一步 | 暂时不做 |
|---|---|---|
| 检索接近，但约束/层级出现有解释的信号 | E0 与预先指定的候选补 seeds 42、123，共新增 4 次完整训练 | 不把所有 parent 都扩成三种子 |
| Perron 的 λ=0.1 可能过弱或过强 | 固定核心配置，补 λ=0.03、0.3；先 340 步筛选，晋级从 VAST 重跑 1360 步 | 不从 340 步 checkpoint 续训冒充统一预算 |
| 锥满足率持续为零、parent 卡下限 | 分析损失贡献、半径和优化；区分权重问题与对齐/锥约束冲突 | 不仅靠加大 λ 或换维度解释为成功 |
| 检索、真实层级均无稳定收益 | 保留 E0，研究无 entailment 的 Perron proxy | 不为找到正结果无限扩大搜索 |

每次补参数都在验证集选择，并保存理由；若给某 parent 更多搜索预算，报告这一差异。
不修改 A 的队列，不用 B 的临时赢家替换 A 的 Gate A 选择。

## 6. P2：proxy 与真实层级评估

这些评估无需新训练，但工具仍需接入；当前仓库没有完整可执行的类别/层级管线。

- E0 原始 Perron proxy 的半径与跨模态分散程度关系、T/V/A 权重分布。
- TV、TA、VA、TVA 子集 proxy 及缺失模态检索。
- ActivityNet、AudioSet 的真实祖先 F1/Hit@k；MSR-VTT 是类别—实例诊断，不能当作多层 ontology。
- 区分类别文本→实例与样本内部 P→T/V/A；锥满足率提高不等于语义层级改善。
- 所有模型使用相同标签、固定提示词、候选集合和评价脚本。

如果层级指标用于选择 parent，必须有独立的层级验证划分。只有测试标签时，结果仅作
冻结配置后的描述性分析，不据此选参。缺少层级结果或训练 seed 波动估计时，Gate B 保持未决。
约束保留条件沿用计划：满足率至少约 30%、验证 Top-50 相对 E0 下降不超过 0.3 个百分点，
且真实层级收益超过测得的 seed 波动。查询 bootstrap 不能替代训练 seed 方差。

## 7. P3：先 64 维，再决定是否 16 维

先做 `M-E0 / M+Perron / PMRL-global64 / hyperbolic allpair` 四组。固定 backbone 和 ITM，
只改变检索嵌入维度及其匹配初始化，模型仍从 VAST 完整训练。

- PCA 基底只用**当前训练划分**拟合；不能默认复用曾看过新 holdout 的旧基底。
- M/allpair 温度候选为 0.03、0.07、0.15；PMRL 的 τ1 候选为 0.05、0.07、0.1，τ2 固定 0.1。
- 首先给 E0、PMRL、allpair 各 3 个 340 步筛选（共 9 个）。验证排序规则运行前固定，
  建议主看 TV/TA Stage-1 的等权均值，Top-50 单独报告。
- Perron 先沿用选定 M-E0 的温度，λ=0.1，保证同温度的 E0/ET 配对。
  如补 0.03/0.3，单独记录额外预算；不能把分别改变温度后的差值归因于 entailment。
- 晋级配置均重新完整训练。先确认 64 维结果稳定，再决定 16 维、Text/SVD 和额外种子。
- 当前维度投影仅支持 64/16；32 维需要实现和验收。现有 `add lowdim` 的清单包含 kernel，
  尚不等同于这里的 PMRL 四组矩阵，不能直接调用并声称完成本方案。

分别回答“低维 M 是否有竞争力”和“固定 M 配置下 entailment 是否有额外收益”。
512 维接近不代表低维一定有优势；已有旧 K 的 64/16 维 parent 结果也不能替代新 M/Perron 对照。

## 8. 交付与回合并检查单

- [ ] 记录 Git commit、冻结源码差异/哈希、环境、初始化哈希、划分 ID 指纹。
- [ ] 保存各 trial 配置、有效更新数、失败/重试记录、训练日志和验证指标。
- [ ] 单独标识 smoke、临时 E0、正式复验及历史旧 K 结果，不混合统计。
- [ ] 汇总 E0 配对差值；单 seed 结果注明探索性；原始和 floored parent 分开分析。
- [ ] 接收 Gate A 结果后，登记配置是否变化，补最终 E0＋候选的复验。
- [ ] 未完成层级工具、低维实现或 Gate A 回退设计时明确记录缺项。
- [ ] 代码/计划通过 Git 共享；数据、权重、日志和模型文件按单独授权的方式传递。
