import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

#The class to perform stereo risk minimization
class MyFunction(Function):

    @staticmethod
    def forward(cost, disp_values):

        # cost: the probability for each disparity hypothesis
        #       shape (B, D, H, W), B is batch size, D is the number of disparity hypothesis, H and W are height and width     
        # disp_values: the disparity values, sorted
        #       shape (B, D, H, W), B is batch size, D is the number of disparity hypothesis, H and W are height and width  
        band = 1.1
        left = disp_values[:, 0:1]
        right = disp_values[:, -1:]
       
        step = 0.0
        while True:
            mid = (left + right) / 2.0
            tau = (mid - disp_values) / band
            der = torch.sign(tau) * (1 - torch.exp(-tau.abs()))
            der = der * cost
            der = der.sum(dim=1, keepdim=True)

            #flag = (der > 0.0).to(left.dtype)
            #left = left * flag + mid * (1-flag)
            #right = mid * flag + right * (1-flag)
            flag = der > 0.0
            left = torch.where(flag, left, mid)
            right = torch.where(flag, mid, right)

            step = step + 1

            if der.abs().max() < 1e-2:
                break

        #print(step, der.abs().max())
        mid = (left + right) / 2.0

        return mid

    @staticmethod
    def setup_context(ctx, inputs, output):
        cost, disp_values = inputs
        ctx.save_for_backward(cost, disp_values, output)

    @staticmethod
    def backward(ctx, grad_output):
        band = 1.1
        cost, disp_values, mid = ctx.saved_tensors
        tau = (mid - disp_values) / band
        etau = torch.exp(-tau.abs())

        der_pi = torch.sign(tau) * (1 - etau)

        der_mid = (cost * (etau / band)).sum(dim=1, keepdim=True)

        der_mid = torch.where(der_mid > 0.1, der_mid, der_mid * 0.0 + 0.1)

        der = - der_pi / der_mid

        der_cost = der * grad_output
        der_disp = None

        return der_cost, der_disp
