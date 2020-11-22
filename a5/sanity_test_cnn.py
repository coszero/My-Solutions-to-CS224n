import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
"""
Sanity test of CNN block.
"""

s_len_b = 10
w_len = 6
e_char = 7
e_word = 4

net = CNN(in_channels=e_char, out_channels=e_word)


x = torch.randn(s_len_b, e_char, w_len)  # shape(s_len_b, e_char, w_len)

print("input size: {}".format(x.size()))
print("input type: {}".format(type(x)))
print("input: {}".format(x))

x_cnn = net.forward(x)

print("output size: {}".format(x_cnn.size()))
print("output type: {}".format(type(x_cnn)))
print("output: {}".format(x_cnn))
