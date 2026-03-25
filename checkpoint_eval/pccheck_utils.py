import torch


def set_storage(model, optimizer_list, gpu_ar):
    """
    将模型参数、梯度和优化器状态映射到 GPU 内存区域。
    
    支持两种模式：
      1. gpu_ar 为单个 tensor（legacy）：按 [param|grad|exp_avg|exp_avg_sq] 布局映射
      2. gpu_ar 为 dict（四块独立分配）：分别映射到各自的 tensor
    
    Args:
        model: PyTorch 模型
        optimizer_list: 优化器列表（通常包含一个 Adam 优化器）
        gpu_ar: 预分配的 CUDA tensor（大小 4×N），或 dict{'param','grad','exp_avg','exp_avg_sq'}
    """
    if isinstance(gpu_ar, dict):
        return _set_storage_streams(model, optimizer_list, gpu_ar)

    start_idx = 0
    model_size = 0
    
    # ==================== Region 1: Model Parameters ====================
    for name, ref in model.named_parameters():
        end_idx = start_idx + ref.numel()
        my_ar = gpu_ar[start_idx:end_idx]
        prev_shape = ref.size()
        with torch.no_grad():
            temp = ref.clone()
            ref.set_(my_ar, 0, tuple(prev_shape))
            ref.copy_(temp)
        start_idx += ref.numel()
        model_size += ref.numel()

    # ==================== Region 2: Gradients ====================
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    end_idx = start_idx + p.grad.numel()
                    my_ar = gpu_ar[start_idx:end_idx]
                    prev_shape = p.grad.size()
                    p.grad.set_(my_ar, 0, tuple(prev_shape))
                    start_idx += p.grad.numel()
    
    # ==================== Region 3 & 4: Optimizer States ====================
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    
                    momentum_key = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                    if momentum_key:
                        exp_avg = state[momentum_key]
                        end_idx = start_idx + exp_avg.numel()
                        my_ar = gpu_ar[start_idx:end_idx]
                        prev_shape = exp_avg.size()
                        with torch.no_grad():
                            temp = exp_avg.clone()
                            exp_avg.set_(my_ar, 0, tuple(prev_shape))
                            exp_avg.copy_(temp)
                        start_idx += exp_avg.numel()
                    
                    variance_key = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                    if variance_key:
                        exp_avg_sq = state[variance_key]
                        end_idx = start_idx + exp_avg_sq.numel()
                        my_ar = gpu_ar[start_idx:end_idx]
                        prev_shape = exp_avg_sq.size()
                        with torch.no_grad():
                            temp = exp_avg_sq.clone()
                            exp_avg_sq.set_(my_ar, 0, tuple(prev_shape))
                            exp_avg_sq.copy_(temp)
                        start_idx += exp_avg_sq.numel()
    
    print(f"✅ [set_storage] gpu_ar 内存映射完成:")
    print(f"   - Model params: [0, {model_size})")
    print(f"   - Gradients: [{model_size}, {model_size*2})")
    print(f"   - exp_avg: [{model_size*2}, {model_size*3})")
    print(f"   - exp_avg_sq: [{model_size*3}, {model_size*4})")
    print(f"   - Total used: {start_idx} / {gpu_ar.numel()} ({100*start_idx/gpu_ar.numel():.1f}%)")
    
    return model_size


def _set_storage_streams(model, optimizer_list, gpu_buffers):
    """将模型状态映射到四块独立的 GPU tensor 上。"""
    buf_param = gpu_buffers['param']
    buf_grad = gpu_buffers['grad']
    buf_exp_avg = gpu_buffers.get('exp_avg')
    buf_exp_avg_sq = gpu_buffers.get('exp_avg_sq')

    model_size = 0
    idx = 0

    for name, ref in model.named_parameters():
        end = idx + ref.numel()
        my_ar = buf_param[idx:end]
        prev_shape = ref.size()
        with torch.no_grad():
            temp = ref.clone()
            ref.set_(my_ar, 0, tuple(prev_shape))
            ref.copy_(temp)
        idx += ref.numel()
        model_size += ref.numel()

    grad_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    end = grad_idx + p.grad.numel()
                    my_ar = buf_grad[grad_idx:end]
                    prev_shape = p.grad.size()
                    p.grad.set_(my_ar, 0, tuple(prev_shape))
                    grad_idx += p.grad.numel()

    ea_idx = 0
    eas_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    mk = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                    if mk and buf_exp_avg is not None:
                        t = state[mk]
                        end = ea_idx + t.numel()
                        my_ar = buf_exp_avg[ea_idx:end]
                        with torch.no_grad():
                            tmp = t.clone()
                            t.set_(my_ar, 0, tuple(t.size()))
                            t.copy_(tmp)
                        ea_idx += t.numel()

                    vk = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                    if vk and buf_exp_avg_sq is not None:
                        t = state[vk]
                        end = eas_idx + t.numel()
                        my_ar = buf_exp_avg_sq[eas_idx:end]
                        with torch.no_grad():
                            tmp = t.clone()
                            t.set_(my_ar, 0, tuple(t.size()))
                            t.copy_(tmp)
                        eas_idx += t.numel()

    total_used = model_size + grad_idx + ea_idx + eas_idx
    total_alloc = sum(b.numel() for b in gpu_buffers.values() if b is not None)
    print(f"✅ [set_storage] 四块独立 GPU 内存映射完成:")
    print(f"   - param  buffer: {buf_param.numel():,} elements")
    print(f"   - grad   buffer: {buf_grad.numel():,} elements")
    print(f"   - exp_avg       : {buf_exp_avg.numel() if buf_exp_avg is not None else 0:,} elements")
    print(f"   - exp_avg_sq    : {buf_exp_avg_sq.numel() if buf_exp_avg_sq is not None else 0:,} elements")
    print(f"   - Total used: {total_used:,} / {total_alloc:,} ({100*total_used/total_alloc:.1f}%)")

    return model_size


def initialize(model, optimizer_list, do_opt_step=True, separate_streams=False):
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.state_dict()

    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        if len(opt_state['state']) == 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = p.data.new(p.size())
        if do_opt_step:
            optimizer.step()

    model_size = 0
    for name, ref in model_state.items():
        if (torch.is_tensor(ref)):
            model_size += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            model_size += 1

    opt_size = 0
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                if torch.is_tensor(ref):
                    opt_size += ref.numel()
    
    total_size = model_size + model_size + opt_size

    if separate_streams:
        gpu_buffers = {
            'param': torch.zeros(model_size, device='cuda'),
            'grad': torch.zeros(model_size, device='cuda'),
            'exp_avg': torch.zeros(model_size, device='cuda'),
            'exp_avg_sq': torch.zeros(model_size, device='cuda'),
        }
        return gpu_buffers, total_size
    else:
        gpu_ar = torch.zeros(total_size).cuda()
        return gpu_ar, total_size


def get_total_size(model, optimizer_list):
    model_state = model.state_dict()
    model_size = 0
    for name, ref in model_state.items():
        if (torch.is_tensor(ref)):
            model_size += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            model_size += 1

    opt_size = 0
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                # print(k, ref.dtype)
                if (torch.is_tensor(ref)):
                    opt_size += ref.numel()
                elif (type(ref) == int or type(ref) == float):
                    opt_size += 1

    return model_size + opt_size
