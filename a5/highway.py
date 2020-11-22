#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):

    """
    Highway networks for character-level embeddings
    Named dimensions:
    - b       = batch size
    - s_len   = length of sentence
    - e_word  = word embedding length
    """

    ### YOUR CODE HERE for part 1f

    def __init__(self, e_word, dropout):

        """
        Creates a highway network to improve word embeddings from character-level
        convolutional neural network.
        @param e_words: (int) size of input word embeddings
        @param dropout: (float) dropout rate used in dropout layer
        """

        super(Highway, self).__init__()

        self.weights_proj = nn.Linear(e_word, e_word)  # weight matrix of x_proj, size(e_words, e_words)
        self.weights_gate = nn.Linear(e_word, e_word)  # weight matrix of x_gate, size(e_words, e_words)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_conv):

        """
        The forward pass of a Highway block that maps input from CNN to highway's output.
        @param x_conv: (torch.Tensor) output from previous layer, with shape(s_len, b, e_words)
        @return: (torch.Tensor) The Highway block output of shape(s_len, b, e_words)
        """

        x_proj = F.relu(self.weights_proj(x_conv))
        x_gate = torch.sigmoid(self.weights_gate(x_conv))

        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1-x_gate), x_conv)
        x_word_emb = self.dropout(x_highway)

        return x_word_emb








    ### END YOUR CODE

