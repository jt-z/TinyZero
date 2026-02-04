# TinyZero训练Resume功能实现说明

## 功能概述

本文档详细说明了TinyZero项目中训练Resume功能的实现原理和逻辑。该功能允许训练在中断后从之前保存的checkpoint自动恢复，无需重新开始训练。

## 实现原理

### 1. Checkpoint检测机制

训练开始时，系统会自动检测是否存在之前保存的checkpoint：

- **检测路径**：`checkpoints/TinyZero/actor/` 目录
- **检测逻辑**：查找所有 `global_step_xxx` 格式的目录，提取step数并找到最大值
- **返回值**：返回最新的checkpoint step数，若不存在则返回None

### 2. 模型加载机制

当检测到checkpoint时，系统会从相应的路径加载模型：

- **Actor模型**：从 `checkpoints/TinyZero/actor/global_step_xxx` 加载
- **Critic模型**：从 `checkpoints/TinyZero/critic/global_step_xxx` 加载
- **Tokenizer**：从checkpoint目录加载对应的tokenizer

### 3. 训练步数恢复机制

系统会根据checkpoint的step数设置训练的起始步数：

- **新训练**：从step 1开始
- **Resume训练**：从checkpoint的step数开始

## 代码修改详解

### 1. RayPPOTrainer类修改

#### 1.1 添加checkpoint检测方法

```python
def _find_latest_checkpoint(self):
    """Find the latest checkpoint step"""
    import os
    import re
    
    checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, 'actor')
    if not os.path.exists(checkpoint_dir):
        return None
    
    # find all global_step directories
    step_pattern = re.compile(r'global_step_(\d+)')
    steps = []
    
    for dir_name in os.listdir(checkpoint_dir):
        match = step_pattern.match(dir_name)
        if match:
            steps.append(int(match.group(1)))
    
    if steps:
        return max(steps)
    return None
```

#### 1.2 修改初始化方法

在 `__init__` 方法末尾添加checkpoint检测：

```python
# check for latest checkpoint
self.latest_checkpoint_step = self._find_latest_checkpoint()
print(f'Found latest checkpoint at step: {self.latest_checkpoint_step}')
```

#### 1.3 修改fit方法

修改 `fit` 方法，支持从checkpoint恢复训练：

```python
# initialize global_steps from latest checkpoint if available
if self.latest_checkpoint_step is not None:
    self.global_steps = self.latest_checkpoint_step
    print(f'Resuming training from step {self.global_steps}')
else:
    self.global_steps = 0

# perform validation before training only if starting from scratch
if self.latest_checkpoint_step is None and self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
    val_metrics = self._validate()
    pprint(f'Initial validation metrics: {val_metrics}')
    logger.log(data=val_metrics, step=self.global_steps)
    if self.config.trainer.get('val_only', False):
        return

# we start from step 1 if no checkpoint, or continue from current step
if self.latest_checkpoint_step is None:
    self.global_steps += 1
```

### 2. ActorRolloutRefWorker类修改

#### 2.1 修改init_model方法

添加checkpoint检测和加载逻辑：

```python
# check for checkpoint
import os
import re
checkpoint_dir = os.path.join('checkpoints/TinyZero/actor')
latest_checkpoint = None

if os.path.exists(checkpoint_dir):
    # find all global_step directories
    step_pattern = re.compile(r'global_step_(\d+)')
    steps = []
    
    for dir_name in os.listdir(checkpoint_dir):
        match = step_pattern.match(dir_name)
        if match:
            steps.append(int(match.group(1)))
    
    if steps:
        latest_step = max(steps)
        latest_checkpoint = os.path.join(checkpoint_dir, f'global_step_{latest_step}')
        print(f'Found latest actor checkpoint: {latest_checkpoint}')

# use latest checkpoint if available
model_path = latest_checkpoint if latest_checkpoint else self.config.model.path
```

### 3. CriticWorker类修改

#### 3.1 修改init_model方法

添加checkpoint检测和加载逻辑：

```python
# check for checkpoint
import os
import re
checkpoint_dir = os.path.join('checkpoints/TinyZero/critic')
latest_checkpoint = None

if os.path.exists(checkpoint_dir):
    # find all global_step directories
    step_pattern = re.compile(r'global_step_(\d+)')
    steps = []
    
    for dir_name in os.listdir(checkpoint_dir):
        match = step_pattern.match(dir_name)
        if match:
            steps.append(int(match.group(1)))
    
    if steps:
        latest_step = max(steps)
        latest_checkpoint = os.path.join(checkpoint_dir, f'global_step_{latest_step}')
        print(f'Found latest critic checkpoint: {latest_checkpoint}')

# pass checkpoint path to _build_critic_model_optimizer
self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
    self.config, checkpoint_path=latest_checkpoint)
```

#### 3.2 修改_build_critic_model_optimizer方法

修改方法签名和实现，支持从checkpoint加载模型：

```python
def _build_critic_model_optimizer(self, config, checkpoint_path=None):
    # the following line is necessary
    import os
    from verl.utils.model import LambdaLayer, print_model_size, squeeze
    from verl.utils.torch_dtypes import PrecisionType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
    from torch import optim

    # use checkpoint path if provided, otherwise use original model path
    if checkpoint_path:
        local_path = checkpoint_path
        print(f'Loading critic model from checkpoint: {local_path}')
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
    
    # use tokenizer from checkpoint if available, otherwise use original tokenizer path
    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, 'tokenizer.json')):
        tokenizer_path = checkpoint_path
    else:
        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        
    self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))
```

## 实现关键点

### 1. 路径管理

- **统一路径**：使用 `self.config.trainer.default_local_dir` 作为基础路径，确保路径的一致性
- **动态构建**：根据checkpoint step动态构建模型和tokenizer的加载路径

### 2. 兼容性处理

- **向后兼容**：当不存在checkpoint时，系统会自动从原始模型路径加载，确保新训练的正常进行
- **路径优先级**：checkpoint路径优先于原始模型路径

### 3. 信息反馈

- **详细日志**：在关键步骤打印详细的日志信息，包括找到的checkpoint路径、加载状态等
- **明确提示**：在恢复训练时打印 "Resuming training from step xxx" 消息，明确告知用户训练状态

## 使用方法

### 启动训练

直接运行训练脚本即可，系统会自动处理checkpoint的检测和加载：

```bash
bash scripts/start_countdown_training.sh
```

### 预期行为

1. **首次训练**：系统会从step 1开始训练，并在每100步保存checkpoint
2. **Resume训练**：系统会自动检测最新的checkpoint并从相应的step开始训练

### 检查训练状态

训练开始时，系统会打印以下信息：

```
Found latest checkpoint at step: 200
Resuming training from step 200
epoch 0, step 200
```

## 注意事项

1. **Checkpoint保存频率**：系统默认每100步保存一次checkpoint，可通过 `trainer.save_freq` 配置调整

2. **内存管理**：Resume训练时的内存使用与新训练相同，需要确保GPU内存充足

3. **训练稳定性**：Resume训练从checkpoint恢复模型权重和训练步数，但不会恢复优化器状态，这可能会导致训练初期的轻微波动

4. **路径配置**：确保 `trainer.default_local_dir` 配置正确，指向 `checkpoints/TinyZero/` 目录

## 结论

通过实现训练Resume功能，TinyZero项目现在能够在训练中断后自动从之前的checkpoint恢复训练，大大提高了训练的可靠性和效率。该实现方案具有良好的兼容性和扩展性，可以适应不同的训练场景。

## 未来优化方向

1. **优化器状态保存**：未来可以考虑保存和恢复优化器状态，进一步提高Resume训练的连续性

2. **Checkpoint压缩**：实现checkpoint的压缩和清理机制，减少存储空间的占用

3. **自动故障恢复**：添加自动故障检测和恢复机制，在检测到OOM等问题时自动调整参数并Resume训练

4. **分布式训练支持**：确保在分布式训练场景下Resume功能的正常工作

---

**作者**：TinyZero开发团队
**日期**：2026-02-05
**版本**：v1.0