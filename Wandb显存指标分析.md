# Wandb 显存指标分析

## 一、显存空余状态指标

### 1. Actor 相关显存空余指标

#### 1.1 批处理结束时的空闲显存
- **指标名**：`actor/memory/free`
- **含义**：Actor 批处理完成后，当前 GPU 上的剩余空闲显存
- **计算方式**：`torch.cuda.get_device_properties(0).total_memory / 1024³ - memory_allocated`
- **重要性**：反映每个训练批次结束后的显存恢复情况

#### 1.2 优化器步骤后的空闲显存
- **指标名**：`actor/optimizer_step_memory/free`
- **含义**：Actor 优化器更新步骤完成后，当前 GPU 上的剩余空闲显存
- **重要性**：监控优化器更新对显存的影响，是最能反映内存使用趋势的指标之一

### 2. Critic 相关显存空余指标

#### 2.1 批处理结束时的空闲显存
- **指标名**：`critic/memory/free`
- **含义**：Critic 批处理完成后，当前 GPU 上的剩余空闲显存
- **重要性**：反映 Critic 网络训练后的显存恢复情况

#### 2.2 优化器步骤后的空闲显存
- **指标名**：`critic/optimizer_step_memory/free`
- **含义**：Critic 优化器更新步骤完成后，当前 GPU 上的剩余空闲显存
- **重要性**：监控 Critic 优化器更新对显存的影响

## 二、显存状态综合分析

除了直接的空闲显存指标外，以下指标也能帮助分析显存状态：

| 指标名 | 含义 | 与空闲显存的关系 |
|--------|------|------------------|
| `actor/memory/allocated` | 当前已分配的显存 | 与空闲显存负相关 |
| `actor/memory/reserved` | 预留的显存 | 反映内存池大小，影响实际可用内存 |
| `actor/memory/model` | 模型参数占用的显存 | 固定开销，不随训练变化 |
| `actor/memory/optimizer` | 优化器状态占用的显存 | 固定开销，不随训练变化 |
| `actor/memory/active` | 激活值占用的显存 | 与 batch size 和序列长度正相关 |
| `actor/memory/other` | 其他内存（如缓存） | 可能随训练累积，影响空闲显存 |

## 三、如何使用这些指标监控显存状态

### 1. 监控趋势
- **关注重点**：`actor/optimizer_step_memory/free` 的变化趋势
- **预警信号**：若该指标持续下降，可能预示内存泄漏
- **正常状态**：在稳定训练中，该指标应保持相对稳定

### 2. 设置阈值
- **安全阈值**：空闲显存应保持在 2GB 以上
- **警告阈值**：空闲显存低于 2GB 时，应警惕 OOM 风险
- **危险阈值**：空闲显存低于 1GB 时，极可能发生 OOM 错误

### 3. 分析瓶颈
- **激活值内存**：`actor/memory/active` 过高可能是序列长度或 batch size 过大
- **其他内存**：`actor/memory/other` 持续增长可能是内存泄漏或缓存累积
- **预留内存**：`actor/memory/reserved` 远大于 `actor/memory/allocated` 可能是内存碎片严重

### 4. 比较阶段
- **批处理前后**：对比 `batch_start` 和 `batch_end` 的空闲显存，评估内存清理效果
- **优化器步骤**：对比 `optimizer_step` 前后的空闲显存，评估优化器更新对内存的影响
- **不同批次**：对比不同训练批次的内存使用，发现内存使用的变化趋势

## 四、代码实现分析

### 1. 内存监控核心函数

**文件**：`verl/utils/debug/memory_monitor.py`

```python
def log_detailed_memory_usage(head, model=None, optimizer=None, active_tensors=None):
    """详细内存监控函数，监控各部分内存使用情况"""
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
    if other_memory < 0:
        other_memory = 0.0
    
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

### 2. 指标集成到 Wandb

**文件**：`verl/workers/actor/dp_actor.py` 和 `verl/workers/critic/dp_critic.py`

```python
# 记录批处理结束时的内存状态
batch_end_memory = log_detailed_memory_usage('After actor update, batch end', 
                               model=self.actor_module, 
                               optimizer=self.actor_optimizer, 
                               logger=logger)

# 记录优化器步骤后的内存状态
memory_metrics = log_detailed_memory_usage('After optimizer step', 
                               model=self.actor_module, 
                               optimizer=self.actor_optimizer, 
                               logger=logger)

# 将内存指标添加到返回的 metrics 字典
for key, value in batch_end_memory.items():
    metrics[f'actor/{key}'] = value

# 添加优化器步骤的内存指标
for key, value in memory_metrics.items():
    metrics[f'actor/optimizer_step_{key}'] = value
```

## 五、显存问题排查指南

### 1. 内存泄漏排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `actor/memory/other` 持续增长 | 中间缓存未清理 | 增加 `torch.cuda.empty_cache()` 调用频率 |
| `actor/optimizer_step_memory/free` 持续下降 | 内存泄漏 | 检查代码中是否有未释放的张量或缓存 |
| `actor/memory/reserved` 远大于 `actor/memory/allocated` | 内存碎片 | 重启训练或使用内存池优化 |

### 2. OOM 错误预防

| 预防措施 | 具体操作 | 效果 |
|----------|----------|------|
| 监控空闲显存 | 设置 `actor/memory/free` 阈值告警 | 提前发现内存不足风险 |
| 调整 batch size | 减小 `ppo_micro_batch_size` | 直接减少内存使用 |
| 调整序列长度 | 减小 `max_prompt_length` 和 `max_response_length` | 减少激活值内存 |
| 启用内存卸载 | 设置 `param_offload=True` 和 `optimizer_offload=True` | 将部分内存转移到 CPU |
| 定期清理缓存 | 在关键节点添加 `torch.cuda.empty_cache()` | 减少内存碎片 |

## 六、实际案例分析

### 案例：训练几十个 step 后 OOM

#### 症状
- `actor/memory/free` 初始正常（约 8-10GB）
- 随着训练进行，该指标逐渐下降
- 几十个 step 后，降至接近 0GB，触发 OOM

#### 原因分析
1. **内存碎片累积**：训练过程中内存分配和释放产生碎片
2. **其他内存增长**：`actor/memory/other` 可能持续增长
3. **预留内存增加**：`actor/memory/reserved` 可能逐渐增大

#### 解决方案
1. **增加缓存清理频率**：在每个 micro batch 后添加缓存清理
2. **优化 FSDP 配置**：考虑启用部分内存卸载
3. **减小 batch size**：降低 `ppo_micro_batch_size` 以减少内存使用
4. **监控其他内存**：分析 `actor/memory/other` 增长的原因

## 七、总结

Wandb 中的显存指标是监控训练健康状况的重要工具，特别是：

1. **`actor/optimizer_step_memory/free`**：最能反映内存使用趋势，是内存监控的核心指标
2. **`actor/memory/other`**：可能指示内存泄漏或缓存累积，需要重点关注
3. **`actor/memory/reserved`**：反映内存碎片情况，与 `allocated` 的差距过大可能需要优化

通过合理设置监控阈值、定期分析内存使用趋势、及时排查内存问题，可以有效预防 OOM 错误，提高训练稳定性和效率。

在 TinyZero 项目中，这些显存指标已经集成到训练流程中，为开发者提供了全面的内存使用可视化，帮助快速定位和解决内存相关问题。