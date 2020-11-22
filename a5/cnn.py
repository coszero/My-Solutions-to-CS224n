#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):

    """
    Convolutional Neural Network to process words in character level.
    Named dimension:
    - bs_len =   batch size * sentence length
    - w_len  =   number of characters in a word
    - e_char =   dimension of character embeddings
    - e_word =   dimension of word embeddings
    """

    ### YOUR CODE HERE for part 1g
    def __init__(self, in_channels, out_channels, kernel_size=5):
        """
        Create a CNN to process words in character level.
        @param in_channels: (int) dimension of character embeddings
        @param out_channels: (int) dimension of word embeddings
        """
        super(CNN, self).__init__()

        self.kernel_size = kernel_size
        self.padding = 1
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels
                              , kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x_reshaped):
        """
        The forward pass of Conv1d block to map input character embedding matrix to conv_out embeddings.
        @param x_reshaped: (torch.Tensor) input character embedding matrix with shape (bs_len, e_char, w_len).
        @return: x_conv_out (torch.Tensor) output embedding matrix processed by CNN block, with shape(bs_len, e_word)
        """

        x_conv = F.relu(self.conv(x_reshaped))
        x_conv_out = torch.max(x_conv, dim=-1).values

        return x_conv_out

    ### END YOUR CODE

