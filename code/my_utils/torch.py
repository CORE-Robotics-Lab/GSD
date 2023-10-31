import torch
import numpy as np
from torch.autograd import Variable
import hashlib

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
device_cpu = torch.device("cpu")

DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor


def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)


def zeros(*shape):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(Variable(zeros(param.data.view(-1).shape)))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(Variable(zeros(param.data.view(-1).shape)))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def upsample(arr, dim0):
    asize = list(arr.shape)
    mulp, rem = dim0 // asize[0], dim0 % asize[0]
    arr1 = torch.tile(arr, [mulp] + [1]*(len(asize)-1))
    arr2 = arr[:rem]
    return torch.cat([arr1, arr2], 0)


def normalize_tens(tens, dim=1, eps=1e-8):
    return tens / (torch.linalg.norm(tens, dim=dim, keepdim=True) + eps)


def normalize_arr(arr, axis=1, eps=1e-8):
    return arr / (np.linalg.norm(arr, axis=axis, keepdims=True) + eps)


def update_linear_schedule(optimizer, progress, initial_lr):
    lr = initial_lr - (initial_lr * progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def nn_init(module, xavier=False, ortho=False, actv='relu'):
    gain = torch.nn.init.calculate_gain(actv)
    if ortho:
        weight_init = torch.nn.init.orthogonal_
        bias_init = lambda x, gain: torch.nn.init.constant_(x, 0)
    elif xavier:
        weight_init = torch.nn.init.xavier_uniform_
        bias_init = lambda x, gain: torch.nn.init.constant_(x, 0)
    else:
        weight_init = lambda x, gain: None
        bias_init = lambda x, gain: None
    weight_init(module.weight.data, gain=gain)
    if hasattr(module.bias, 'data'): bias_init(module.bias.data, gain=gain)
    return module


def tensorhash(tens):
    tens1 = tens.data.numpy()
    return hashlib.sha256(tens1.data).hexdigest()[:10], str(tens1.sum()), str(tens1.max()), str(tens1.min())
