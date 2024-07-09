# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:02:31 2024

@author: Lenovo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CCLoss(nn.Module):
    def __init__(self):
        super(CCLoss, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.padding = 0

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3),
                               stride=(1, 1), padding=(0, 0),dilation=(1,1))
        
        
    def forward(self, image, target):
        #bce = F.binary_cross_entropy_with_logits(image, target)
        I2 = torch.mul(image,image)
        J2 = torch.mul(target,target)
        IJ = torch.mul(image, target)
        sum_filter = torch.ones([9,9,1,1])
        I_sum = self.conv1(image)       
        J_sum = self.conv1(target) 
        I2_sum = self.conv1(I2)
        J2_sum = self.conv1(J2)
        IJ_sum = self.conv1(IJ)
        
        win_size = 9*9
        
        u_I = I_sum/win_size
        u_J = J_sum/win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
        
        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        
        return -1.0*torch.mean(cc)


'''
import tensorflow as tf
import numpy as np

def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss
'''