# 小维度实验：新服务器交接

更新：2026-10-19。按最新要求，使用 **GitHub 传输代码和轻量标注，在新服务器重建缓存、继续测试和调试**。不需要也不建议通过 SSH 搬运当前生成的媒体缓存。交接分支为 `lowdim-handoff-20261019`；不包含大文件传输。

## 1. 接手后先知道这些

实验配置已经确定：MSRVTT + ActivityNet Captions；VAST、GRAM、PMRL、双曲有锥/无锥、欧氏有锥/无锥共 7 个方法；维度 8/32/128/512；seed=50；共 **56 次端到端微调**。所有方法都更新三个编码器，10 epochs，全局 batch=64。新的 512D 也要重跑，不能用旧结果替代。

- 当前未运行任何本轮 CUDA 任务、PCA 特征提取、正式训练；没有小维度实验结果。
- 当前 11 个 CPU 单元/运行时检查通过（含新增无缓存 bootstrap 测试）：几何、投影初始化、全局采样、排名、模拟 checkpoint 提交中断及恢复、显存规划等。
- 已通过 CPU/Gloo 两进程固定输入对照：四个自有版本和三个原生基线的 D8/D512 损失与梯度。原生 ITM/DAM 使用固定负样本及简化编码器夹具，不能当作完整 CUDA 模型验证。
- 真实 VAST/GRAM/PMRL 和七子集模型已成功在 CPU 构造、加载权重，原生投影头形状和四个自有版本的同初始化已核查。随后追加的完整 CPU 前向检查在切换为交接任务时中止，**不要标记为已全部完成**。
- 原服务器 MSRVTT 8,811 个视频缓存已完成，冻结 train/val/test 为 7427/500/884；音频抽样与原 AudioMapper 逐元素相同。ActivityNet 最后日志为 **3400/12067、失败 0**，尚未完成。
- 已按最新要求停止本任务未完成的缓存和 CPU 验证进程，保留已产生文件；没有停止其他人的进程。
- **未生成 `CPU_READY.json` / `GPU_READY.json`**。后续交接入口和就绪检查代码有更新，新服务器应重新跑完整验证。CPU 通过不等于 CUDA 数值、显存、吞吐或断点恢复已通过。

本包是待新服务器完成验证的实现，不是已跑完的实验产物。不要绕过就绪门槛或手工 touch READY 文件。

## 2. GitHub 应传什么

交接目录可作为独立 GitHub 仓库的根目录，避免把原工作区混在一起提交：

```text
README.md                  本文
code/                      本项目必要的 model/utils/data、lowdim 入口及测试
  requirements-lowdim-finetune.txt
  experiments/subset_hierarchy/LOWDIM_README.md  完整实验协议和运行细节
authors/VAST/              原生基线所需代码及 LICENSE
authors/GRAM/              原生基线所需代码及 LICENSE
annotations/               固定候选 train/val/test 视频 ID 和 captions
protocol.json, tasks.json  参数快照和 56 项任务表
SHA256.json                包内文件校验
.gitignore
```

**不包含**：视频、音频、帧/fbank/token 缓存、模型权重、虚拟环境、运行日志、CPU/GPU 就绪门槛。少量 BPE tokenizer `.gz` 是代码运行必需的小文件，已包含。

原工作区还有此前的未提交修改。不要从原工作区执行 `git add .` 来迁移；应只审核并提交这个独立目录。该目录单独发布到原仓库的交接分支，主分支保持不变。

## 3. 新服务器必须已经有或另行取得的大文件

GitHub 不负责传输以下资源。在目标服务器使用当地现有资源或原始下载来源：

| 资源 | 精确要求 |
|---|---|
| MSRVTT 视频/音频 | 候选集合 8811 个 ID；视频 `<ID>.mp4`，音频 `<ID>.wav` |
| ActivityNet 视频/音频 | 候选集合 12067 个 ID；同样文件命名；不要另换 split |
| VAST 初始化权重 | `model_step_204994.pt`，SHA256 `9005d283ea0de99549efbedeb7d21c8e25bea8fa41c0f6570296b0e7a055a6c5` |
| GRAM 初始化权重 | `gram_model_step_459.pt`，SHA256 `070c4b25a12f0a2861ced77d9bbe871e649f9be5b390e680e6049114a51b99d5` |
| PMRL 初始化权重 | `pmrl_base.pt`，SHA256 `78ab07cf45f9465fcd1cb0a01b7c32ec83a467e642338a67bd6079bdb21b8836` |
| 编码器构造资源 | 一个 assets 目录，结构见下 |

```text
assets/
  clip/EVA01_CLIP_g_14_psz14_s11B.pt
  beats/BEATs_iter3_plus_AS2M.pt
  bert/bert-base-uncased/
    config.json
    pytorch_model.bin
    vocab.txt
    tokenizer_config.json 等原有 tokenizer 配置
```

权重 SHA 必须一致。文件名相似不代表可以替换。若新服务器没有这些确切权重，先记录阻塞，确认来源/取得方式；不要静默换模型。若媒体路径、格式、音轨或解码结果不同，先核实，不能用删样本掩盖差异。

## 4. 从 GitHub 代码启动，不依赖旧缓存

以下路径都是示例，必须改为目标服务器当前用户有权读写的路径。不修改系统 Python、共享 HOME 或其他人的任务。使用项目内环境，保持 `PYTHONNOUSERSITE=1`；先检查磁盘，再预处理。

```bash
# 在下载的独立交接仓库根目录
repo_root="$PWD"
data_root=/your/authorized/data
suite_root="$data_root/lowdim_finetune_20261019"

cd "$repo_root/code"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="$repo_root/code/.cache/pip"
export HF_HOME="$repo_root/code/.cache/huggingface"
export TORCH_HOME="$repo_root/code/.cache/torch"
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
# 先确认目标驱动/GPU和合适的 PyTorch CUDA wheel。
# requirements 中记录原环境的版本，若调整 PyTorch 则必须重新验证。
.venv/bin/python -m pip install -r requirements-lowdim-finetune.txt
python_bin="$repo_root/code/.venv/bin/python"

ROOT_ARGS=(--workspace "$repo_root/code" --author-root "$repo_root/authors" \
  --data-root "$data_root" --suite "$suite_root")

# 首先把轻量候选标注和模型配置映射到新服务器。
# 所有媒体/权重/assets 路径须在 data_root 或 workspace 下。
"$python_bin" -B experiments/subset_hierarchy/lowdim_handoff.py bootstrap \
  "${ROOT_ARGS[@]}" --package "$repo_root" \
  --msrvtt-video "$data_root/msrvtt/videos" --msrvtt-audio "$data_root/msrvtt/wav" \
  --activitynet-video "$data_root/activitynet/videos" --activitynet-audio "$data_root/activitynet/wav" \
  --vast-checkpoint "$data_root/weights/model_step_204994.pt" \
  --gram-checkpoint "$data_root/weights/gram_model_step_459.pt" \
  --pmrl-checkpoint "$data_root/weights/pmrl_base.pt" \
  --assets "$data_root/assets"

# 本机重建缓存；先少量样本，再全量。
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_data.py --stage cache "${ROOT_ARGS[@]}" --workers 2 --limit 4
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_data.py --stage cache "${ROOT_ARGS[@]}" --workers 8

# 全量缓存完成后，在本机逐阶段验证并生成新的 CPU_READY。
# 不需要调用 export，也不需要搬运新服务器生成的缓存。
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_verify.py --stage tests "${ROOT_ARGS[@]}"
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_verify.py --stage models "${ROOT_ARGS[@]}"
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_verify.py --stage cache "${ROOT_ARGS[@]}"
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_verify.py --stage seal "${ROOT_ARGS[@]}"
```

`bootstrap` 只检查路径和所需媒体是否存在，真正解码检查在 cache 阶段，权重 SHA 在 models 阶段。缺媒体时 bootstrap 明确失败，不自动重算交集。新服务器不要执行旧的 `lowdim_data.py --stage prepare`，该入口依赖原服务器历史实验；请使用上述 bootstrap。

候选 ActivityNet train/val/test=7580/500/3987。全量解码后如有少量失败会写 `excluded.json`，较大失败率直接停止；务必核查实际 READY 的计数。默认字幕/标题均沿用候选 captions。

## 5. GPU 阶段：留待新服务器检查资源后显式运行

先确认授权使用的 GPU 和空闲显存，不结束未知任务。所有命令仍带相同 `ROOT_ARGS`。下面以 8 卡举例；也可用有授权的更小列表。

```bash
"$python_bin" -B experiments/subset_hierarchy/lowdim_schedule.py "${ROOT_ARGS[@]}"  # 只看状态
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage basis --execute-gpu --devices 0,1,2,3,4,5,6,7 "${ROOT_ARGS[@]}"
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage profile --execute-gpu --devices 0,1,2,3,4,5,6,7 "${ROOT_ARGS[@]}"
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage preflight --execute-gpu --devices 0,1,2,3,4,5,6,7 "${ROOT_ARGS[@]}"
"$python_bin" -u -B experiments/subset_hierarchy/lowdim_schedule.py --stage train --execute-gpu --devices 0,1,2,3,4,5,6,7 "${ROOT_ARGS[@]}"
```

- basis：各数据集、各原生模型提取 train-only 512D 特征，形成 6 个嵌套 PCA 基；D512 保持恒等。
- profile：每个数据集和模型家族尝试 world size 1/2/4/8；至少 2 次 optimizer step 后计时；只有两份峰值加 3 GiB 能放下且双任务吞吐提高 ≥10%，才启用同卡两任务。
- preflight：7 方法 × 2 维度（8/512）× 2 数据集，共 28 次完整 CUDA smoke，检查数值、编码器/投影头非零梯度、共享初始化。通过后封存 GPU_READY。
- train：按实测布局运行/恢复 56 项任务，全局 batch 不缩水。同一个 run 固定 world size，恢复 optimizer/scaler/各 rank RNG。每卡至多两个本任务 rank、全局至多 16 个。调度和完整 CUDA 路径尚未在当前服务器实测，首次跑必须查看日志。

## 6. 接手者优先检查/调试清单

1. 先跑 `bootstrap`，确认媒体和三个权重的真实来源、文件布局与 SHA。
2. 重建少量缓存，重点检查目标服务器音频解码后通道/采样率是否与本协议一致；再跑全量。
3. 跑完整 CPU 验证。Gloo 需要本机进程通信，当前原服务器沙箱曾阻止 loopback；这是执行权限问题，不是损失差异。不得因此删掉分布式检查。
4. 完成 GPU basis，随后 profile；重点看原生 ITM/DAM、DDP 非重入 checkpoint、BEATs LayerDrop、FP16 初始缩放、完整编码器显存。
5. 完成 28 smokes 再正式训练；训练恢复涉及大文件/NFS，原服务器只做过小 CPU 状态的提交中断测试，仍需在目标设备实际验证。
6. 若代码修改，重新生成对应 CPU/GPU 门槛；不要用旧结果文件或旧 `DONE` 冒充当前源码通过。

## 7. 代码入口及结果

- `lowdim_handoff.py`：无缓存 bootstrap 和本地 GitHub 交接目录导出。
- `lowdim_data.py`：数据预处理、冻结 cohort、确定性输入、统一全局采样。
- `lowdim_models.py`：模型加载、train-only PCA 投影、四个自有版本共享初始化。
- `lowdim_finetune.py`：DDP、验证选模、训练/测试、断点保存和恢复。
- `lowdim_schedule.py`：显式 GPU 调度、profile、smoke 和门槛。
- `lowdim_verify.py`：CPU 检查及就绪记录。
- `lowdim_report.py`：56 项状态、六方向 R@1/5/10、macro、相对新 512D 保持率、几何指标；可选 matplotlib PDF/PNG。
- `utils/lowdim_geometry.py`：本轮欧氏/双曲锥损失、PCA、评估。

```bash
"$python_bin" -B experiments/subset_hierarchy/lowdim_report.py "${ROOT_ARGS[@]}"
# 可选安装 matplotlib 后加 --plots；训练本身不需要 matplotlib。
```

所有表和图注明单 seed，不声称跨 seed 稳健性；未完成的格子必须保留 PENDING/FAILED/INCOMPLETE，不借用历史结果。更完整的数学、训练和缓存细节见 `code/experiments/subset_hierarchy/LOWDIM_README.md`。
