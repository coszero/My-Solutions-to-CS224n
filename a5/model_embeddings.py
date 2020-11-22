#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    Named dimensions:
    - b      =   batch size
    - s_len  =   number of words in a sentence
    - w_len  =   number of characters in a word
    - e_word =   dimension of word embeddings
    - e_char =   dimension of character embeddings
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        pad_token_idx = vocab.char2id['<pad>']

        self.vocab = vocab
        self.word_embed_size = word_embed_size
        self.dropout = 0.3
        self.e_char = 50
        self.cnn = CNN(in_channels=self.e_char, out_channels=self.word_embed_size)
        self.highway = Highway(e_word=self.word_embed_size, dropout=self.dropout)
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.vocab.char2id), embedding_dim=self.e_char, padding_idx=pad_token_idx)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        input_emb = self.embedding_layer(input)  # (s_len, b, w_len, e_char)
        input_emb_reshaped = input_emb.view(-1, input_emb.size(2), input_emb.size(3))\
            .permute(0, 2, 1)  # (s_len*b, e_char, w_len)
        x_conv_out = self.cnn(input_emb_reshaped).view(input_emb.size(0),
                                                       input_emb.size(1), -1)  # (s_len, b, e_word)
        x_emb = self.highway(x_conv_out)  # (s_len, b, e_word)

        return x_emb


        ### END YOUR CODE

