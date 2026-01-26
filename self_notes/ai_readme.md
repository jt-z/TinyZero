# AI 助手动交互记录

本文档记录了 AI 助手在 `TinyZero` 项目中的操作和与用户的交互。

## 初始任务：熟悉项目并调试

1.  **任务**: 我首先阅读了 `README.md` 和您在 `self_notes` 中记录的 `操作记录.md`，了解到项目目标以及之前遇到的 `libcuda.so` 链接错误和 CUDA OOM (内存溢出) 问题。
2.  **提交早期修改**: 根据您的要求，我将一系列文件和修改提交到了 Git 仓库。这包括：
    *   为调试脚本添加的 `echo` 启动信息。
    *   封装了环境变量的 `start_countdown_training.sh` 启动脚本。
    *   您记录的 `self_notes` 目录下的所有文件 (`操作记录.md`, `cuda_test_script.py`, `wandb_test_script.py`, `ai_readme.md`)。
3.  **解决 Git 推送问题**: 在此期间，我们遇到了 HTTPS 协议的认证失败问题。根据您的指示，我指导您将远程仓库 URL 切换为 SSH 格式，最终成功解决了推送问题。

## `train_tiny_zero.sh` 的参数迭代优化

这个阶段的核心目标是找到一组能在您的硬件上稳定运行并解决潜在 OOM 问题的训练参数。我们对 `scripts/train_tiny_zero.sh` 文件进行了一系列连续的修改和实验：

1.  **修复 NCCL 冲突**:
    *   在 `start_countdown_training.sh` 中添加了 `NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1` 来解决潜在的 P2P 通信问题。

2.  **架构级修复 (禁用张量并行)**:
    *   将 `tensor_model_parallel_size` 从 `2` 改为 `1`，让每个 GPU 独立运行模型，从根本上避免跨 GPU 通信问题。
    *   将 `gpu_memory_utilization` 调整为 `0.5`。

3.  **24GB 显存初步优化**:
    *   将 `gpu_memory_utilization` 进一步降低到 `0.2`。
    *   将 `ppo_micro_batch_size` 和 `log_prob_micro_batch_size` 从 `8` 改为 `4`，以减少瞬时显存峰值。

4.  **高级显存优化与 CPU Offload**:
    *   添加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来优化 PyTorch 的内存分配。
    *   添加了 `RAY_memory_monitor_refresh_ms=0` 来防止 Ray 因瞬时内存波动而误杀进程。
    *   启用了 FSDP 的 `optimizer_offload`，将优化器状态卸载到 CPU，以节省显存。

5.  **参数的反复权衡和微调**:
    *   **"黄金平衡"模式**: 尝试了 `gpu_memory_utilization=0.15`，并将模型参数保留在 GPU 上（`param_offload=False`），同时增大了全局 Micro Batch Size。
    *   **"极致平衡"模式**: 将全局 Micro Batch Size 调回 `8`（单卡为 `1`），寻求最稳定的配置。
    *   **"All-In GPU + 长度裁剪"模式**: 关闭了所有 Offload，并将 `max_response_length` 从 `1024` 减少到 `896`，作为最后的保险措施。
    *   **"vLLM=0.1 + Optimizer Offload Only"模式**: 重新启用了 `optimizer_offload`，并将 `max_response_length` 恢复到 `1024`。

## 最终配置

*   **当前配置**: 我们最终确定了 `"Final: All-In GPU, No Offload"` 配置。该配置旨在将所有计算和参数都保留在 GPU 上，不使用任何 Offload 技术，以获取最直接的性能表现，同时通过调整 `gpu_memory_utilization` 和 Batch Size 来确保稳定性。

**总结**: 我们的尝试是一个典型的模型训练调优过程：从解决环境和硬件的底层问题开始，然后通过不断调整模型并行策略、显存使用率、Batch Size 大小以及 FSDP 的各种 Offload 策略，在**性能**和**稳定性**之间寻找最佳平衡点。