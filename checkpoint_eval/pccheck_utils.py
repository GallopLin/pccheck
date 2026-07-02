import torch


def _tensor_matches_buffer_view(tensor, buffer_view):
    if tensor is None or buffer_view is None:
        return False
    if not torch.is_tensor(tensor) or not torch.is_tensor(buffer_view):
        return False
    if tensor.numel() != buffer_view.numel():
        return False
    if tensor.device != buffer_view.device or tensor.dtype != buffer_view.dtype:
        return False
    return tensor.data_ptr() == buffer_view.data_ptr()


def _model_param_numel(model):
    return sum(ref.numel() for _, ref in model.named_parameters())


def _grad_numel(optimizer_list):
    total = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total += p.grad.numel()
    return total


def _reserved_grad_numel(model, optimizer_list):
    # legacy PCCheck layout 默认为 [param|grad|exp_avg|exp_avg_sq] 四段连续区间。
    # 即使当前一步结束后 p.grad 被 DeepSpeed 释放，也显式预留一整段 grad 零区，
    # 这样 checkpoint 格式与恢复侧的偏移假设保持一致。
    actual_grad_size = _grad_numel(optimizer_list)
    return max(actual_grad_size, _model_param_numel(model))


def _optimizer_tensor_numel(optimizer_list):
    total = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p, {})
                momentum_key = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                if momentum_key:
                    total += state[momentum_key].numel()
                variance_key = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                if variance_key:
                    total += state[variance_key].numel()
    return total


def _is_single_buffer_layout_mapped(model, optimizer_list, gpu_ar):
    if gpu_ar is None or gpu_ar.numel() == 0:
        return False

    offset = 0
    matched_any = False

    for _, ref in model.named_parameters():
        end = offset + ref.numel()
        if end > gpu_ar.numel():
            return False
        if not _tensor_matches_buffer_view(ref, gpu_ar[offset:end]):
            return False
        matched_any = True
        offset = end

    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    end = offset + p.grad.numel()
                    if end > gpu_ar.numel():
                        return False
                    if not _tensor_matches_buffer_view(p.grad, gpu_ar[offset:end]):
                        return False
                    offset = end

    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p not in optimizer.state:
                    continue

                state = optimizer.state[p]
                momentum_key = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                if momentum_key:
                    t = state[momentum_key]
                    end = offset + t.numel()
                    if end > gpu_ar.numel():
                        return False
                    if not _tensor_matches_buffer_view(t, gpu_ar[offset:end]):
                        return False
                    offset = end

                variance_key = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                if variance_key:
                    t = state[variance_key]
                    end = offset + t.numel()
                    if end > gpu_ar.numel():
                        return False
                    if not _tensor_matches_buffer_view(t, gpu_ar[offset:end]):
                        return False
                    offset = end

    return matched_any


def _is_stream_buffer_layout_mapped(model, optimizer_list, gpu_buffers):
    buf_param = gpu_buffers.get('param')
    buf_grad = gpu_buffers.get('grad')
    buf_exp_avg = gpu_buffers.get('exp_avg')
    buf_exp_avg_sq = gpu_buffers.get('exp_avg_sq')

    if buf_param is None or buf_param.numel() == 0:
        return False

    matched_any = False
    param_idx = 0
    for _, ref in model.named_parameters():
        end = param_idx + ref.numel()
        if end > buf_param.numel():
            return False
        if not _tensor_matches_buffer_view(ref, buf_param[param_idx:end]):
            return False
        matched_any = True
        param_idx = end

    grad_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if buf_grad is None:
                        return False
                    end = grad_idx + p.grad.numel()
                    if end > buf_grad.numel():
                        return False
                    if not _tensor_matches_buffer_view(p.grad, buf_grad[grad_idx:end]):
                        return False
                    grad_idx = end

    exp_avg_idx = 0
    exp_avg_sq_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p not in optimizer.state:
                    continue

                state = optimizer.state[p]
                momentum_key = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                if momentum_key:
                    if buf_exp_avg is None:
                        return False
                    t = state[momentum_key]
                    end = exp_avg_idx + t.numel()
                    if end > buf_exp_avg.numel():
                        return False
                    if not _tensor_matches_buffer_view(t, buf_exp_avg[exp_avg_idx:end]):
                        return False
                    exp_avg_idx = end

                variance_key = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                if variance_key:
                    if buf_exp_avg_sq is None:
                        return False
                    t = state[variance_key]
                    end = exp_avg_sq_idx + t.numel()
                    if end > buf_exp_avg_sq.numel():
                        return False
                    if not _tensor_matches_buffer_view(t, buf_exp_avg_sq[exp_avg_sq_idx:end]):
                        return False
                    exp_avg_sq_idx = end

    return matched_any


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
        if _is_stream_buffer_layout_mapped(model, optimizer_list, gpu_ar):
            model_size = _model_param_numel(model)
            print("✅ [set_storage] 检测到 storage 已映射到独立 GPU buffers，跳过重复 remap")
            return model_size
        return _set_storage_streams(model, optimizer_list, gpu_ar)

    if _is_single_buffer_layout_mapped(model, optimizer_list, gpu_ar):
        model_size = _model_param_numel(model)
        print("✅ [set_storage] 检测到 storage 已映射到 gpu_ar，跳过重复 remap")
        return model_size

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

    def _check_capacity(buffer, start, length, label):
        if buffer is None:
            raise RuntimeError(f"[set_storage] missing buffer for {label}")
        end = start + length
        if end > buffer.numel():
            raise RuntimeError(
                f"[set_storage] {label} buffer is too small: need end={end:,}, "
                f"capacity={buffer.numel():,}. This usually means initialize() "
                "computed buffer sizes before optimizer states were created."
            )

    model_size = 0
    idx = 0

    for name, ref in model.named_parameters():
        end = idx + ref.numel()
        _check_capacity(buf_param, idx, ref.numel(), f"param:{name}")
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
                    _check_capacity(buf_grad, grad_idx, p.grad.numel(), "grad")
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
                        _check_capacity(buf_exp_avg, ea_idx, t.numel(), mk)
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
                        _check_capacity(buf_exp_avg_sq, eas_idx, t.numel(), vk)
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
                    # 使用零梯度，这样 optimizer.step() 不会改变参数值
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
        if do_opt_step:
            optimizer.step()
            # 重置 step 计数器，避免影响后续训练的学习率调度
            for state in optimizer.state.values():
                if 'step' in state:
                    if isinstance(state['step'], torch.Tensor):
                        state['step'].zero_()
                    else:
                        state['step'] = 0

    param_size = _model_param_numel(model)
    actual_grad_size = _grad_numel(optimizer_list)
    reserved_grad_size = _reserved_grad_numel(model, optimizer_list)
    opt_size = _optimizer_tensor_numel(optimizer_list)
    total_size = param_size + reserved_grad_size + opt_size

    print(
        f"[initialize] param_size={param_size}, live_grad_size={actual_grad_size}, "
        f"reserved_grad_size={reserved_grad_size}, opt_state_size={opt_size}, total_size={total_size}"
    )

    if separate_streams:

        # 确保优化器状态已初始化，这样 exp_avg/exp_avg_sq 才能被正确映射。
        # 用零梯度执行一次 step：grad=0 时参数不更新，但会创建 Adam 状态张量。
        already_initialized = any(len(opt.state) > 0 for opt in optimizer_list)
        if not already_initialized:
            for optimizer in optimizer_list:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
            for optimizer in optimizer_list:
                optimizer.step()
            # 重置 step 计数器，避免影响后续训练
            for optimizer in optimizer_list:
                for state in optimizer.state.values():
                    if 'step' in state:
                        if isinstance(state['step'], torch.Tensor):
                            state['step'].zero_()
                        else:
                            state['step'] = 0

        actual_grad_size = _grad_numel(optimizer_list)
        reserved_grad_size = _reserved_grad_numel(model, optimizer_list)
        opt_size = _optimizer_tensor_numel(optimizer_list)
        total_size = param_size + reserved_grad_size + opt_size
        print(
            f"[initialize] separate_streams updated sizes after optimizer state init: "
            f"live_grad_size={actual_grad_size}, reserved_grad_size={reserved_grad_size}, "
            f"opt_state_size={opt_size}, total_size={total_size}"
        )

        exp_avg_size = opt_size // 2
        exp_avg_sq_size = opt_size - exp_avg_size
        gpu_buffers = _initialize_streams_memory_saving(
            model,
            optimizer_list,
            param_size=param_size,
            reserved_grad_size=reserved_grad_size,
            exp_avg_size=exp_avg_size,
            exp_avg_sq_size=exp_avg_sq_size,
        )
        actual_total = param_size + reserved_grad_size + exp_avg_size + exp_avg_sq_size
        return gpu_buffers, actual_total
    else:
        gpu_ar = _initialize_memory_saving(
            model,
            optimizer_list,
            total_size,
            reserved_grad_size=reserved_grad_size,
        )
        return gpu_ar, total_size


def _cpu_pin_copy(tensor):
    cpu_tensor = tensor.detach().cpu()
    try:
        return cpu_tensor.pin_memory()
    except RuntimeError:
        return cpu_tensor


def _initialize_streams_memory_saving(
    model,
    optimizer_list,
    param_size,
    reserved_grad_size,
    exp_avg_size,
    exp_avg_sq_size,
):
    """低峰值四块流式布局初始化，避免旧状态和新 buffers 同时驻留 GPU。"""
    import gc

    param_cpu_list = []
    for name, ref in model.named_parameters():
        param_cpu_list.append((name, ref.size(), _cpu_pin_copy(ref.data)))

    grad_cpu_list = []
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    grad_cpu_list.append(None)
                else:
                    grad_cpu_list.append((p.grad.size(), _cpu_pin_copy(p.grad.data)))

    opt_cpu_data = []
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if not state:
                    opt_cpu_data.append(None)
                    continue

                entry = {}
                momentum_key = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                if momentum_key:
                    t = state[momentum_key]
                    entry['mk'] = momentum_key
                    entry['exp_avg'] = (t.size(), _cpu_pin_copy(t.data))

                variance_key = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                if variance_key:
                    t = state[variance_key]
                    entry['vk'] = variance_key
                    entry['exp_avg_sq'] = (t.size(), _cpu_pin_copy(t.data))

                step = state.get('step', None)
                if torch.is_tensor(step):
                    entry['step'] = step.detach().cpu().clone()
                else:
                    entry['step'] = step
                opt_cpu_data.append(entry)

    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state:
                    for k in list(state.keys()):
                        if torch.is_tensor(state[k]) and state[k].is_cuda:
                            state[k] = torch.empty(0, device='cuda')

    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = None

    for _, ref in model.named_parameters():
        ref.data = torch.empty(0, device='cuda')

    gc.collect()
    torch.cuda.empty_cache()
    total = param_size + reserved_grad_size + exp_avg_size + exp_avg_sq_size
    print(
        f"[_initialize_streams_memory_saving] GPU old tensors released; "
        f"allocating stream buffers total={total:,} floats ({total * 4 / 1024**3:.2f} GiB)"
    )

    gpu_buffers = {
        'param': torch.zeros(param_size, dtype=torch.float32, device='cuda'),
        'grad': torch.zeros(reserved_grad_size, dtype=torch.float32, device='cuda'),
        'exp_avg': torch.zeros(max(exp_avg_size, 1), dtype=torch.float32, device='cuda')[:exp_avg_size],
        'exp_avg_sq': torch.zeros(max(exp_avg_sq_size, 1), dtype=torch.float32, device='cuda')[:exp_avg_sq_size],
    }

    param_idx = 0
    model_size = 0
    for (name, orig_shape, cpu_data), (_, ref) in zip(param_cpu_list, model.named_parameters()):
        numel = cpu_data.numel()
        end = param_idx + numel
        my_ar = gpu_buffers['param'][param_idx:end]
        my_ar.copy_(cpu_data.view(-1))
        with torch.no_grad():
            ref.set_(my_ar, 0, tuple(orig_shape))
        param_idx = end
        model_size += numel
    del param_cpu_list

    grad_idx = 0
    grad_entry_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                entry = grad_cpu_list[grad_entry_idx]
                grad_entry_idx += 1
                if entry is None:
                    continue
                orig_shape, cpu_data = entry
                numel = cpu_data.numel()
                end = grad_idx + numel
                if end > reserved_grad_size:
                    raise RuntimeError(
                        f"Gradient stream overflow: need {end:,}, reserved {reserved_grad_size:,}"
                    )
                my_ar = gpu_buffers['grad'][grad_idx:end]
                my_ar.copy_(cpu_data.view(-1))
                p.grad = my_ar.view(orig_shape)
                grad_idx = end
    del grad_cpu_list

    ea_idx = 0
    eas_idx = 0
    opt_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                entry = opt_cpu_data[opt_idx]
                opt_idx += 1
                if entry is None:
                    continue
                state = optimizer.state[p]

                if entry.get('step') is not None:
                    state['step'] = entry['step']

                if 'exp_avg' in entry:
                    mk = entry['mk']
                    orig_shape, cpu_data = entry['exp_avg']
                    numel = cpu_data.numel()
                    end = ea_idx + numel
                    if end > exp_avg_size:
                        raise RuntimeError(
                            f"exp_avg stream overflow: need {end:,}, reserved {exp_avg_size:,}"
                        )
                    my_ar = gpu_buffers['exp_avg'][ea_idx:end]
                    my_ar.copy_(cpu_data.view(-1))
                    state[mk] = my_ar.view(orig_shape)
                    ea_idx = end

                if 'exp_avg_sq' in entry:
                    vk = entry['vk']
                    orig_shape, cpu_data = entry['exp_avg_sq']
                    numel = cpu_data.numel()
                    end = eas_idx + numel
                    if end > exp_avg_sq_size:
                        raise RuntimeError(
                            f"exp_avg_sq stream overflow: need {end:,}, reserved {exp_avg_sq_size:,}"
                        )
                    my_ar = gpu_buffers['exp_avg_sq'][eas_idx:end]
                    my_ar.copy_(cpu_data.view(-1))
                    state[vk] = my_ar.view(orig_shape)
                    eas_idx = end
    del opt_cpu_data
    gc.collect()

    print(f"[_initialize_streams_memory_saving] stream buffers mapped:")
    print(f"   - param     : {model_size:,} / {gpu_buffers['param'].numel():,}")
    print(f"   - grad      : {grad_idx:,} / {gpu_buffers['grad'].numel():,}")
    print(f"   - exp_avg   : {ea_idx:,} / {gpu_buffers['exp_avg'].numel():,}")
    print(f"   - exp_avg_sq: {eas_idx:,} / {gpu_buffers['exp_avg_sq'].numel():,}")

    return gpu_buffers


def _initialize_memory_saving(model, optimizer_list, total_size, reserved_grad_size=None):
    """
    内存优化版 initialize：通过 CPU 暂存避免 GPU 峰值内存翻倍。

    原始流程的问题：
      gpu_ar = torch.zeros(total_size).cuda()   # 需要 ~40 GiB
      但此时 GPU 上已有 param + grad + opt_states（~40 GiB），
      峰值 = 40 + 40 = 80 GiB → OOM on A800-80GB。

    优化流程：
      1. 将 param/grad/opt_states 数据暂存到 CPU pinned memory
      2. 释放 GPU 上的旧 tensor 存储
      3. torch.cuda.empty_cache() 回收碎片
      4. 分配 gpu_ar（此时 GPU 几乎为空）
      5. 从 CPU 拷贝回 gpu_ar 对应区域，用 set_() 重映射

    峰值内存 ≈ gpu_ar 大小（~40 GiB），不再翻倍。
    """
    import gc

    # =====================================================================
    # Stage 1: 将所有需要映射的数据暂存到 CPU pinned memory
    # =====================================================================
    # 1a. Model parameters → CPU
    param_cpu_list = []
    for name, ref in model.named_parameters():
        param_cpu_list.append((ref.size(), ref.data.cpu().pin_memory()))

    # 1b. Gradients → CPU（梯度是零值，只需记录 shape）
    grad_shapes = []
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_shapes.append(p.grad.size())
                else:
                    grad_shapes.append(None)

    # 1c. Optimizer states → CPU
    opt_cpu_data = []
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    entry = {}
                    momentum_key = 'exp_avg' if 'exp_avg' in state else ('next_m' if 'next_m' in state else None)
                    if momentum_key:
                        entry['mk'] = momentum_key
                        entry['exp_avg'] = (state[momentum_key].size(), state[momentum_key].data.cpu().pin_memory())
                    variance_key = 'exp_avg_sq' if 'exp_avg_sq' in state else ('next_v' if 'next_v' in state else None)
                    if variance_key:
                        entry['vk'] = variance_key
                        entry['exp_avg_sq'] = (state[variance_key].size(), state[variance_key].data.cpu().pin_memory())
                    # 保存 step 等标量状态（不映射到 gpu_ar，但需要保留）
                    entry['step'] = state.get('step', None)
                    opt_cpu_data.append(entry)
                else:
                    opt_cpu_data.append(None)

    # =====================================================================
    # Stage 2: 释放 GPU 上的旧 tensor 存储
    # =====================================================================
    # 2a. 释放优化器状态的 GPU 存储
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for k in list(state.keys()):
                        if torch.is_tensor(state[k]) and state[k].is_cuda:
                            # 用空 CUDA tensor 替换，释放显存但保持设备一致
                            state[k] = torch.empty(0, device='cuda')

    # 2b. 释放梯度的 GPU 存储
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = None

    # 2c. 释放参数的 GPU 存储（用空 CUDA tensor 替换，保持设备一致，
    #     否则后续 set_() 映射到 CUDA gpu_ar 时会因设备不匹配报错）
    for name, ref in model.named_parameters():
        ref.data = torch.empty(0, device='cuda')

    # =====================================================================
    # Stage 3: 回收 GPU 碎片
    # =====================================================================
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[_initialize_memory_saving] GPU 已释放旧 tensor，"
          f"准备分配 gpu_ar (total_size={total_size}, "
          f"{total_size * 4 / 1024**3:.2f} GiB)")

    # =====================================================================
    # Stage 4: 分配 gpu_ar（此时 GPU 几乎为空，不会 OOM）
    # =====================================================================
    gpu_ar = torch.zeros(total_size, dtype=torch.float32, device='cuda')

    # =====================================================================
    # Stage 5: 从 CPU 拷贝回 gpu_ar 对应区域，并用 set_() 重映射
    # =====================================================================
    if reserved_grad_size is None:
        reserved_grad_size = _reserved_grad_numel(model, optimizer_list)

    start_idx = 0
    model_size = 0

    # 5a. Region 1: Model Parameters
    for (orig_shape, cpu_data), (name, ref) in zip(param_cpu_list, model.named_parameters()):
        numel = cpu_data.numel()
        end_idx = start_idx + numel
        my_ar = gpu_ar[start_idx:end_idx]
        # 拷贝 CPU 数据到 gpu_ar 切片
        my_ar.copy_(cpu_data.view(-1))
        # 重映射参数存储到 gpu_ar 切片
        with torch.no_grad():
            ref.set_(my_ar, 0, tuple(orig_shape))
        start_idx = end_idx
        model_size += numel

    # 释放 CPU 暂存
    del param_cpu_list

    # 5b. Region 2: Gradients（保持 legacy 布局：即使当前没有 live grad，也显式预留零区）
    grad_region_start = start_idx
    grad_region_end = grad_region_start + reserved_grad_size
    grad_offset = 0
    grad_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                shape = grad_shapes[grad_idx]
                grad_idx += 1
                if shape is not None:
                    numel = 1
                    for s in shape:
                        numel *= s
                    end_idx = grad_region_start + grad_offset + numel
                    if end_idx > grad_region_end:
                        raise RuntimeError(
                            f"Gradient region overflow: need {end_idx - grad_region_start}, reserved {reserved_grad_size}"
                        )
                    my_ar = gpu_ar[grad_region_start + grad_offset:end_idx]
                    # gpu_ar 已经是零，直接映射
                    p.grad = my_ar.view(shape)
                    grad_offset += numel

    start_idx = grad_region_end
    del grad_shapes

    # 5c. Region 3 & 4: Optimizer States
    opt_idx = 0
    for optimizer in optimizer_list:
        for group in optimizer.param_groups:
            for p in group['params']:
                entry = opt_cpu_data[opt_idx]
                opt_idx += 1
                if entry is None:
                    continue
                state = optimizer.state[p]

                # 恢复 step 等标量状态
                if entry.get('step') is not None:
                    state['step'] = entry['step']

                if 'exp_avg' in entry:
                    mk = entry['mk']
                    orig_shape, cpu_data = entry['exp_avg']
                    numel = cpu_data.numel()
                    end_idx = start_idx + numel
                    my_ar = gpu_ar[start_idx:end_idx]
                    my_ar.copy_(cpu_data.view(-1))
                    state[mk] = my_ar.view(orig_shape)
                    start_idx = end_idx

                if 'exp_avg_sq' in entry:
                    vk = entry['vk']
                    orig_shape, cpu_data = entry['exp_avg_sq']
                    numel = cpu_data.numel()
                    end_idx = start_idx + numel
                    my_ar = gpu_ar[start_idx:end_idx]
                    my_ar.copy_(cpu_data.view(-1))
                    state[vk] = my_ar.view(orig_shape)
                    start_idx = end_idx

    del opt_cpu_data
    gc.collect()

    print(f"[_initialize_memory_saving] gpu_ar 分配与映射完成:")
    print(f"   - Model params: [0, {model_size})")
    print(f"   - Grad region : [{model_size}, {model_size + reserved_grad_size}) (live={grad_offset}, reserved={reserved_grad_size})")
    print(f"   - Total used: {start_idx} / {gpu_ar.numel()} ({100*start_idx/gpu_ar.numel():.1f}%)")

    return gpu_ar


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
