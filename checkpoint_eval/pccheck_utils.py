import torch


def set_storage(model, optimizer_list, gpu_ar):
    """
    将模型参数、梯度和优化器状态映射到 gpu_ar 的连续内存区域
    
    🔥 Phase 1 改进：支持完整的优化器状态映射
    
    内存布局（4 × model_size for Adam）：
    ┌─────────────────────────────────────────────────────────────┐
    │ [0, N)         : Model Parameters                           │
    │ [N, 2N)        : Gradients                                  │
    │ [2N, 3N)       : exp_avg (momentum for Adam)                │
    │ [3N, 4N)       : exp_avg_sq (adaptive LR for Adam)          │
    └─────────────────────────────────────────────────────────────┘
    
    Args:
        model: PyTorch 模型
        optimizer_list: 优化器列表（通常包含一个 Adam 优化器）
        gpu_ar: 预分配的 CUDA tensor，大小为 4 × model_size
    """
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
    
    # ==================== Region 3 & 4: Optimizer States (exp_avg, exp_avg_sq) ====================
    # 🔥 Phase 1 新增：映射优化器状态到 gpu_ar
    # 支持多种优化器命名：
    #   - 标准 Adam/AdamW: exp_avg, exp_avg_sq
    #   - BertAdam: next_m, next_v
    for optimizer in optimizer_list:
        # 按参数顺序映射优化器状态
        for group in optimizer.param_groups:
            for p in group['params']:
                # 🔥 注意：optimizer.state 的键是 tensor 对象本身，不是 id(p)
                if p in optimizer.state:
                    state = optimizer.state[p]
                    
                    # 映射 exp_avg / next_m（momentum）
                    # 支持标准 Adam (exp_avg) 和 BertAdam (next_m)
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
                    
                    # 映射 exp_avg_sq / next_v（adaptive learning rate）
                    # 支持标准 Adam (exp_avg_sq) 和 BertAdam (next_v)
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
    
    return model_size  # 🔥 返回 model_size 用于后续计算偏移


def initialize(model, optimizer_list, do_opt_step=True):
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.state_dict()

    # initialize optimizer for realistic setups
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        if len(opt_state['state']) == 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = p.data.new(p.size())
        if do_opt_step:
            optimizer.step()

    # 🔥 Phase 1: 计算 model_size（仅参数）
    model_size = 0
    for name, ref in model_state.items():
        if (torch.is_tensor(ref)):
            model_size += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            model_size += 1

    # 🔥 Phase 1: 计算 opt_size（仅 tensor 状态，不包括标量 step 等）
    # Adam 优化器有：exp_avg, exp_avg_sq（每个参数各一份）
    # 布局：[params | grads | exp_avg | exp_avg_sq] = 4 × model_size
    opt_size = 0
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                # 🔥 只计算 tensor（exp_avg, exp_avg_sq），跳过标量（step）
                if torch.is_tensor(ref):
                    opt_size += ref.numel()
                # 注意：不再计算 int/float，因为它们不需要映射到 gpu_ar
    
    # 🔥 Phase 1: total_size = params + grads + optimizer_tensors
    # 对于 Adam：= N + N + 2N = 4N
    total_size = model_size + model_size + opt_size  # params + grads + opt_tensors
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
