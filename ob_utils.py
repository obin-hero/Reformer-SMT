import os
import numpy as np
import cv2
#def dict_list_to_list_dict():

import time
def time_debug(prev_time, log):
    print("[TIME] ", log, " %.3f"%(time.time() - prev_timme))

def visualize_tensor(tensor, method = 'plt'):
    # assume tensor shape B * C * H * W
    new_tensor = tensor.detach()
    if tensor.device.type == 'cuda' : new_tensor = new_tensor.cpu()
    if len(tensor.shape) == 3 : new_tensor = new_tensor.unsqueeze(0)

    new_tensor = new_tensor.transpose(1,3).numpy() # B * W * H * C
    if new_tensor.shape[-1] != 3:
        new_tensor = np.mean(new_tensor,axis=-1)
    #print(new_tensor.shape)
    view_img = np.concatenate(new_tensor, axis=1)
    #print(view_img.shape)
    if method == 'plt':
        plt.imshow(view_img)
        plt.show()
    elif method == 'array':
        return view_img
    elif method == 'cv2':
        cv2.imshow('hi', view_img[:,:,[2,1,0]])
        cv2.waitKey(0)

def visualize_tensor_with_action(tensor, action, method='cv2'):
    new_tensor = tensor.detach() # tensor B T C H W
    B = tensor.shape[0]
    if tensor.device.type == 'cuda' : new_tensor = new_tensor.cpu()
    if len(tensor.shape) == 3 : new_tensor = new_tensor.unsqueeze(0)
    new_tensor = new_tensor.transpose(2,4).split(1,0)
    new_tensor = torch.cat(new_tensor, 3).numpy() # T * (W*B) * H * C
    if new_tensor.shape[-1] != 3:
        new_tensor = np.mean(new_tensor,axis=-1)
    if len(new_tensor.shape) > 4:
        new_tensor = new_tensor.squeeze(0)
    if method == 'cv2':
        T = new_tensor.shape[0]
        for t in range(T):
            img = cv2.resize(new_tensor[t], (256*B, 256))
            for b in range(B):
                act_t = torch.argmax(action[b][t])
                if act_t == 1: act_t = 1
                elif act_t == 2: act_t = -1
                cv2.line(img, (256*b+128, 256), (int(256*b+128 - 40 * act_t), 256 - 40), (255, 0, 0), 3)
            if img.shape[-1] == 3:
                cv2.imshow('visualize tensor with act', img[:,:,[2,1,0]])
            else:
                cv2.imshow('visualize tensor with act', img)
            cv2.waitKey(0)
    elif method == 'arrray':
        return new_tensor

def mkdir(path):
    prev_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(prev_path) and len(prev_path) > 0:
        mkdir(prev_path)
    if not os.path.exists(path): os.mkdir(path)

def calc_log_losses(loss_criterion, ret='dict', index_multiply=True):
    value = [0] * len(loss_criterion[0])
    num = [0] * len(loss_criterion[0])
    for line_loss in loss_criterion:  # prob_loss_t
        for i, l in enumerate(line_loss):
            value[i] += l.detach().numpy()
            num[i] += 1
    arr = np.array(value) / np.array(num)
    if index_multiply:
        dic = {str(i): value * (i + 1) for i, value in enumerate(arr)}
    else:
        dic = {str(i): value for i, value in enumerate(arr)}
    dic.update({'total': np.mean(arr)})
    if ret == 'dict':
        return dic
    else:
        return arr

import matplotlib.pyplot as plt
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    passed_params = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                passed_params.append(n)
                continue
            #if 'Memory' in n:
            #    print(n, p.grad.abs().mean())
            #    print(n, p.grad.abs().max())
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    #print(passed_params)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.1)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

    plt.show()

import torch
def load_torch_network(net, dir_path, epoch_to_load=None, return_detail=False):
    pretrained_list = sorted([a for a in os.listdir(dir_path) if '.pt' in a])
    latest_pth = pretrained_list[-1] if len(pretrained_list) > 0 else None
    if latest_pth is not None:
        if epoch_to_load is not None:
            valid = [p for p in pretrained_list if 'ep%03dstep'%(epoch_to_load) in p]
            if len(valid) > 0 : latest_pth = valid[-1]
        epoch = int(latest_pth[latest_pth.find('ep') + 2:latest_pth.find('ep') + 5]) if 'ep' in latest_pth else None
        step = 0
        if 'step' in latest_pth:
            step = int(latest_pth[latest_pth.find('step') + 4:latest_pth.find('step') + 12]) if 'step' in latest_pth else None
        sd = torch.load(os.path.join(dir_path, latest_pth))
        net.load_state_dict(sd)
        print('Loaded :', latest_pth, latest_pth[latest_pth.find('ep') + 2:latest_pth.find('ep') + 5])
        if hasattr(net, 'scheduler'): net.scheduler.last_epoch = step
    else:
        FileNotFoundError("No Pretrained model in ", dir_path)
    if return_detail: return net, epoch, step
    else: return net