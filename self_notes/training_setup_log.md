# 训练设置操作日志

## 目标
根据用户提供的 `Qwen/Qwen2.5-3B-Instruct` 模型（3B参数）和8张3090 GPU的硬件配置，启动项目的训练流程。

## 操作步骤

### 1. 初步分析与README查阅
*   用户希望进行模型训练，但未明确指定具体任务和数据集。
*   查阅项目根目录下的 `README.md` 文件，发现项目专注于"Countdown"和"Multiplication"任务的RL训练。
*   `README.md` 中提供了针对 `Qwen-2.5-3B-Instruct` 模型的特定数据准备和训练指令，确定使用"Countdown task"作为训练任务。

### 2. 创建数据集存储目录
*   为了存放数据预处理后的结果，创建了新的目录：
    ```bash
    mkdir -p dataset/countdown_instruct_data
    ```

### 3. 数据预处理
*   根据 `README.md` 中的指令，针对 `qwen-instruct` 模型运行数据预处理脚本：
    ```bash
    conda run -n zero python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir dataset/countdown_instruct_data
    ```
*   **遇到的问题与解决**：
    *   首次运行脚本时，出现 `ModuleNotFoundError: No module named 'datasets'`。
    *   检查 `requirements.txt` 发现 `datasets` 已在列表中，但提示 `conda activate zero` 失败。
    *   意识到是 `conda` 环境未正确激活或初始化。
    *   改用 `conda run -n zero <command>` 方式，确保命令在 `zero` 环境中执行。
    *   重新执行数据预处理脚本后，成功生成了 `train.parquet` 和 `test.parquet` 文件。

### 4. 准备训练配置
*   检查训练脚本 `./scripts/train_tiny_zero.sh`，确认其通过环境变量接收参数：`N_GPUS`, `BASE_MODEL`, `DATA_DIR`, `ROLLOUT_TP_SIZE`, `EXPERIMENT_NAME`。
*   根据用户需求和 `README.md` 建议，确定以下环境变量值：
    *   `N_GPUS=8` (用户提供)
    *   `BASE_MODEL=/home/ksa/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct` (用户提供)
    *   `DATA_DIR=dataset/countdown_instruct_data` (数据预处理结果)
    *   `ROLLOUT_TP_SIZE=2` (`README.md` 针对3B+模型的建议)
    *   `EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct-8gpu` (自定义描述性名称)
    *   `VLLM_ATTENTION_BACKEND=XFORMERS` (`README.md` 建议)

### 5. 启动训练
*   使用 `conda run` 在 `zero` 环境中启动训练脚本：
    ```bash
    export N_GPUS=8
    export BASE_MODEL=/home/ksa/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct
    export DATA_DIR=dataset/countdown_instruct_data
    export ROLLOUT_TP_SIZE=2
    export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct-8gpu
    export VLLM_ATTENTION_BACKEND=XFORMERS
    conda run -n zero bash ./scripts/train_tiny_zero.sh
    ```

### 6. 训练启动后遇到的问题
*   训练脚本开始执行，但最终因 `wandb.errors.errors.UsageError: No API key configured. Use 'wandb login' to log in.` 错误而中止。
*   **需要用户干预**：用户需要登录 Weights & Biases (W&B) 以提供 API Key，以便训练过程中的日志记录功能正常工作。
*   同时注意到了一些警告信息，如 `NCCL WARN Failed to find CUDA library libcuda.so` 和 `Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes`，这些可能需要在 W&B 问题解决后进一步关注。

### 7. 再次尝试训练及新问题
*   在用户解决 `wandb` 登录问题后，再次启动训练，`wandb` 日志已正常工作。
*   然而，训练过程在模型初始化阶段再次中止，报错信息为 `torch._dynamo.exc.BackendCompilerFailed`，其中包含了 `/usr/bin/ld: cannot find -lcuda: No such file or directory`。
*   **问题分析**：这个错误表明系统在编译 CUDA 相关代码（可能由 Triton 或 `torch.compile` 使用）时，无法找到 CUDA 库 `libcuda.so`。这通常是由于 CUDA Toolkit 未正确安装、`libcuda.so` 文件缺失或其路径未添加到系统 `LD_LIBRARY_PATH` 环境变量中导致的系统级配置问题。
*   **解决方案**：
    1.  **确认 CUDA Toolkit 安装**：用户需要确保系统上已正确安装了与 PyTorch 和 Triton 兼容的 CUDA Toolkit 版本。
    2.  **检查 `libcuda.so` 存在性**：确认 `libcuda.so` 文件存在于 CUDA 安装目录（通常在 `/usr/local/cuda/lib64` 或类似路径）。
    3.  **配置 `LD_LIBRARY_PATH`**：确保 `LD_LIBRARY_PATH` 环境变量包含 `libcuda.so` 所在的目录。例如：
        ```bash
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        ```
        （请根据实际 CUDA 安装路径进行调整）

### 8. 当前训练状态和日志总结

**当前训练结果概述:**

目前还没有实际的训练结果可以展示（例如模型的损失、准确率等）。由于系统级别的 CUDA 配置问题，训练过程尚未成功启动。

**主要日志和遇到的问题:**

1.  **数据预处理成功:**
    *   已成功为 "Countdown" 任务生成了训练数据，并存储在 `dataset/countdown_instruct_data` 目录中。

2.  **Weights & Biases (wandb) 登录问题已解决:**
    *   `wandb` 的认证问题已解决，日志系统现在可以正常工作。

3.  **核心阻塞问题：CUDA 库链接错误 (`-lcuda not found`)**
    *   在尝试启动训练时，系统报告无法找到 `libcuda.so` 库。这导致了 `torch._dynamo.exc.BackendCompilerFailed` 错误，训练无法进行。
    *   这个错误表明您的系统 CUDA 环境配置不完整，可能缺少 `libcuda.so` 文件，或者其路径未正确添加到 `LD_LIBRARY_PATH` 环境变量中。

**结论:**

为了能够顺利进行训练，首要任务是解决 CUDA 环境的配置问题。详细的错误信息和建议的解决方案已记录在上述“7. 再次尝试训练及新问题”部分。一旦 CUDA 环境问题解决，我们就可以再次尝试启动训练，届时将会有实际的训练结果和 `wandb` 上的详细日志。

