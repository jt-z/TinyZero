# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
import logging
import gc


def get_model_memory_usage(model):
    """计算模型参数占用的内存"""
    total_params = 0
    for param in model.parameters():
        if param.data.is_cuda:
            total_params += param.data.numel() * param.data.element_size()
    return total_params / 1024**3  # GB


def get_optimizer_memory_usage(optimizer):
    """计算优化器状态占用的内存"""
    total_memory = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None and p.grad.is_cuda:
                total_memory += p.grad.numel() * p.grad.element_size()
            # AdamW 等优化器有额外状态
            if p in optimizer.state:
                state = optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        total_memory += value.numel() * value.element_size()
    return total_memory / 1024**3  # GB


def get_tensor_memory_usage(tensors):
    """计算张量列表占用的内存"""
    total_memory = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            total_memory += tensor.numel() * tensor.element_size()
    return total_memory / 1024**3  # GB


def log_detailed_memory_usage(
    head: str,
    model=None,
    optimizer=None,
    active_tensors=None,
    logger: logging.Logger = None,
    level=logging.DEBUG,
    rank: int = 0
):
    """详细内存监控函数，监控各部分内存使用情况"""
    # 基础内存信息
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_allocated
    
    # 各部分内存信息
    model_memory = 0.0
    optimizer_memory = 0.0
    active_memory = 0.0
    
    if model is not None:
        model_memory = get_model_memory_usage(model)
    
    if optimizer is not None:
        optimizer_memory = get_optimizer_memory_usage(optimizer)
    
    if active_tensors is not None:
        active_memory = get_tensor_memory_usage(active_tensors)
    
    # 计算其他内存（中间缓存等）
    other_memory = memory_allocated - model_memory - optimizer_memory - active_memory
    if other_memory < 0:
        other_memory = 0.0
    
    # 构建消息
    message = (
        f'{head}\n'
        f'  Total: allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB, free={memory_free:.2f}GB\n'
        f'  Model parameters: {model_memory:.2f}GB\n'
        f'  Optimizer states: {optimizer_memory:.2f}GB\n'
        f'  Active tensors (activations): {active_memory:.2f}GB\n'
        f'  Other (caches, etc.): {other_memory:.2f}GB'
    )
    
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)
    
    # 返回内存使用情况字典，用于wandb记录
    return {
        'memory/allocated': memory_allocated,
        'memory/reserved': memory_reserved,
        'memory/free': memory_free,
        'memory/model': model_memory,
        'memory/optimizer': optimizer_memory,
        'memory/active': active_memory,
        'memory/other': other_memory
    }


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    """原有的内存监控函数，保持兼容性"""
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)
