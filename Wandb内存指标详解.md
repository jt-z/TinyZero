# Wandb 内存指标详解

## 一、内存指标概览

TinyZero 训练过程中，wandb 会跟踪大量内存相关指标，这些指标分布在 `actor` 和 `critic` 命名空间下。本文档详细解释每个指标的含义、计算方法以及变化规律。

## 二、内存监控实现原理

### 核心监控函数

内存监控通过 `log_detailed_memory_usage` 函数实现，位于 `verl/utils/debug/memory_monitor.py`：

```python
def log_detailed_memory_usage(head, model=None, optimizer=None, active_tensors=None):
    # 基础内存信息
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_allocated
    
    # 各部分内存信息
    model_memory = get_model_memory_usage(model) if model else 0.0
    optimizer_memory = get_optimizer_memory_usage(optimizer) if optimizer else 0.0
    active_memory = get_tensor_memory_usage(active_tensors) if active_tensors else 0.0
    
    # 计算其他内存
    other_memory = memory_allocated - model_memory - optimizer_memory - active_memory
    
    # 返回内存使用情况字典
    return {
        'memory/allocated': memory_allocated,
        'memory/reserved': memory_reserved,
        'memory/free': memory_free,
        'memory/model': model_memory,
        'memory/optimizer': optimizer_memory,
        'memory/active': active_memory,
        'memory/other': other_memory
    }
```

### 指标记录位置

内存指标在以下两个文件中被记录到 wandb：
- `verl/workers/actor/dp_actor.py`：记录 actor 相关内存指标
- `verl/workers/critic/dp_critic.py`：记录 critic 相关内存指标

## 三、Actor 内存指标详解

### 1. 批处理结束内存指标 (`actor/memory/*`)

这些指标记录在每个批处理结束时的内存状态：

| 指标名称 | 含义 | 计算方法 | 变化规律 |
|---------|------|---------|----------|
| `actor/memory/allocated` | 当前实际分配的 CUDA 内存 | `torch.cuda.memory_allocated() / 1024³` | 相对稳定，随 batch 大小变化 |
| `actor/memory/reserved` | CUDA 上下文预留的内存 | `torch.cuda.memory_reserved() / 1024³` | 可能随 step 增加而增长 |
| `actor/memory/free` | 剩余可用 CUDA 内存 | `总内存 - allocated` | 与 allocated 相反变化 |
| `actor/memory/model` | 模型参数占用的内存 | 所有模型参数张量大小之和 | 训练过程中保持稳定 |
| `actor/memory/optimizer` | 优化器状态占用的内存 | 优化器状态张量大小之和 | 训练过程中保持稳定 |
| `actor/memory/active` | 激活值等活动张量内存 | 活动张量大小之和 | 随序列长度、batch 大小变化 |
| `actor/memory/other` | 其他内存（如缓存） | `allocated - model - optimizer - active` | 可能随 step 增加而累积 |

### 2. 优化器步骤内存指标 (`actor/optimizer_step_memory/*`)

这些指标记录在优化器更新步骤后的内存状态：

| 指标名称 | 含义 | 计算方法 | 变化规律 |
|---------|------|---------|----------|
| `actor/optimizer_step_memory/allocated` | 优化器步骤后实际分配的内存 | `torch.cuda.memory_allocated() / 1024³` | 相对稳定 |
| `actor/optimizer_step_memory/reserved` | 优化器步骤后预留的内存 | `torch.cuda.memory_reserved() / 1024³` | **最可能随 step 增加而增长** |
| `actor/optimizer_step_memory/free` | 优化器步骤后剩余内存 | `总内存 - allocated` | 与 allocated 相反 |
| `actor/optimizer_step_memory/model` | 优化器步骤后模型参数内存 | 所有模型参数张量大小之和 | 稳定 |
| `actor/optimizer_step_memory/optimizer` | 优化器步骤后优化器状态内存 | 优化器状态张量大小之和 | 稳定 |
| `actor/optimizer_step_memory/active` | 优化器步骤后活动张量内存 | 活动张量大小之和 | 随 batch 变化 |
| `actor/optimizer_step_memory/other` | 优化器步骤后其他内存 | `allocated - model - optimizer - active` | 可能累积 |

## 四、Critic 内存指标详解

Critic 部分的内存指标与 Actor 类似，前缀为 `critic/`：

| 指标名称 | 含义 | 变化规律 |
|---------|------|----------|
| `critic/memory/allocated` | Critic 批处理结束时实际分配的内存 | 相对稳定 |
| `critic/memory/reserved` | Critic 批处理结束时预留的内存 | 可能随 step 增长 |
| `critic/memory/free` | Critic 批处理结束时剩余内存 | 与 allocated 相反 |
| `critic/memory/model` | Critic 模型参数内存 | 稳定 |
| `critic/memory/optimizer` | Critic 优化器状态内存 | 稳定 |
| `critic/memory/active` | Critic 活动张量内存 | 随 batch 变化 |
| `critic/memory/other` | Critic 其他内存 | 可能累积 |

## 五、哪些指标会随 step 变化

### 1. 主要变化指标

| 指标名称 | 变化原因 | 变化趋势 |
|---------|---------|----------|
| `actor/optimizer_step_memory/reserved` | **最显著** | 可能逐渐增长 |
| `critic/memory/reserved` | 内存碎片、缓存累积 | 可能逐渐增长 |
| `actor/memory/other` | 中间缓存、内存碎片 | 可能逐渐增长 |
| `critic/memory/other` | 中间缓存、内存碎片 | 可能逐渐增长 |

### 2. 相对稳定指标

| 指标名称 | 稳定原因 |
|---------|---------|
| `actor/memory/model` | 模型参数大小固定 |
| `actor/memory/optimizer` | 优化器状态大小固定 |
| `critic/memory/model` | 模型参数大小固定 |
| `critic/memory/optimizer` | 优化器状态大小固定 |

### 3. 随 batch 变化指标

| 指标名称 | 变化原因 |
|---------|---------|
| `actor/memory/active` | 与 batch 大小、序列长度相关 |
| `actor/memory/allocated` | 与激活值内存相关 |
| `critic/memory/active` | 与 batch 大小、序列长度相关 |
| `critic/memory/allocated` | 与激活值内存相关 |

## 六、内存指标变化分析

### 1. `actor/optimizer_step_memory/reserved` 增长原因

用户观察到只有 `actor/optimizer_step_memory/reserved` 随 step 变化，这是因为：

- **内存碎片**：CUDA 内存分配和释放过程中产生的碎片
- **缓存累积**：训练过程中可能累积的中间缓存
- **PyTorch 内存管理**：PyTorch 的内存池机制可能导致 reserved 内存逐渐增长
- **VLLM 影响**：VLLM 推理引擎可能有内存泄漏或缓存累积

### 2. 内存增长的影响

- **轻微增长**：正常现象，不影响训练
- **快速增长**：可能导致 OOM 错误
- **稳定在安全范围**：理想状态

### 3. 监控建议

1. **关注 `reserved` 内存**：这是最敏感的指标，反映内存管理健康状况
2. **监控 `other` 内存**：可能指示内存泄漏
3. **比较 `allocated` 和 `reserved`**：两者差距过大可能意味着内存碎片严重
4. **结合 GPU 利用率**：内存使用与计算利用率的平衡

## 七、内存优化参考指标

### 1. 理想内存分布

| 组件 | 内存占比（3B模型） | 说明 |
|------|-------------------|------|
| 模型参数 | ~25% | bfloat16 精度 |
| 优化器状态 | ~25% | AdamW 优化器 |
| 激活值 | ~30% | 合理 batch 大小 |
| 其他 | ~20% | 缓存、中间结果 |

### 2. 内存预警阈值

| 指标 | 警告阈值 | 危险阈值 |
|------|---------|----------|
| `memory/reserved` | >80% 总内存 | >90% 总内存 |
| `memory/other` | >30% 总内存 | >40% 总内存 |
| `memory/free` | <2GB | <1GB |

## 八、内存问题排查指南

### 1. 内存泄漏排查

1. **监控 `other` 内存**：持续增长可能指示内存泄漏
2. **检查 `reserved` 内存**：增长过快可能有问题
3. **对比不同 step 的内存分布**：识别异常增长的组件

### 2. 内存优化建议

1. **调整 batch 大小**：减少 `active` 内存
2. **使用内存高效优化器**：如 AdamW8bit 减少 `optimizer` 内存
3. **启用梯度 checkpointing**：减少 `active` 内存
4. **定期清理缓存**：使用 `torch.cuda.empty_cache()`
5. **优化模型并行策略**：如增加 TP 减少每卡内存

### 3. 常见内存问题及解决方案

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 内存碎片 | `reserved` 远大于 `allocated` | 重启训练、使用内存池优化 |
| 激活值爆炸 | `active` 内存过高 | 减少 batch 大小、序列长度 |
| 优化器内存过高 | `optimizer` 内存过高 | 使用 AdamW8bit、Lion 优化器 |
| VLLM 内存泄漏 | `other` 内存持续增长 | 降低 `gpu_memory_utilization` |

## 九、总结

Wandb 中的内存指标是监控训练健康状况的重要工具，特别是：

1. **`actor/optimizer_step_memory/reserved`**：最能反映内存管理状况的指标
2. **`memory/other`**：可能指示内存泄漏或缓存累积
3. **`memory/free`**：直接反映剩余可用内存

通过定期监控这些指标，及时调整训练配置，可以有效避免 OOM 错误，提高训练稳定性和效率。

### 关键监控点

- **训练开始**：建立内存基准线
- **训练中期**：监控内存增长趋势
- **训练后期**：确保内存稳定在安全范围

合理的内存管理是成功训练大模型的关键因素之一，通过本文档的指导，您可以更好地理解和优化 TInyZero 的内存使用。