import torch
import torch.nn as nn
import torch.nn.functional as F
from highway import Highway
"""
Sanity test of Highway block.
"""


net = Highway(e_word=3, dropout=0.5)

x = torch.tensor([[[1., 2., 3.], [5., 6., 7.]],
                  [[3., 5., 6.], [4., 5., 6.]]])  # shape(b, s_len, e_words) (2, 2, 3)

print("input size: {}".format(x.size()))
print("input type: {}".format(type(x)))
print("input: {}".format(x))

x_high = net.forward(x)

print("output size: {}".format(x_high.size()))
print("output type: {}".format(type(x_high)))
print("output: {}".format(x_high))
