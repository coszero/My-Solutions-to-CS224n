#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        Named dimensions:
        - l      =    length of word
        - b      =    batch size
        - h      =    hidden size
        - e_char =    dimensions of character embeddings
        - v_char =    size of character vovab
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.


        char_emb = self.decoderCharEmb(input)  # (l, b, e_char)
        output, dec_hidden = self.charDecoder(char_emb, dec_hidden)  # (l,b,h), [(1,b,h), (1,b,h)]
        scores = self.char_output_projection(output)  # (l,b,v_char)

        return scores, dec_hidden

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        inputs = char_sequence[:-1]  # Remove <end> token
        scores, _ = self.forward(inputs, dec_hidden)  # (l-1,b,v_char)
        scores = scores.permute(1, 2, 0)  # (b, v_char, l-1)
        targets = char_sequence[1:].permute(1, 0)  # (b, l-1)
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        sum_loss = loss(scores, targets)

        return sum_loss


        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        dec_hidden = initialStates
        batch_size = initialStates[0].size(1)
        START, END = self.target_vocab.start_of_word, self.target_vocab.end_of_word
        output_word = [""] * batch_size  # [b,]
        output_word_ind = [[START] * batch_size]  # (1,b)
        current_char_indices = torch.tensor(output_word_ind, device=device)
        for t in range(max_length):
            scores, dec_hidden = self.forward(current_char_indices, dec_hidden)  # (1,b,v_char) [(1,b,h), (1,b,h)]
            scores_softmax = torch.softmax(scores, dim=2)  # (1,b,v_char)
            current_char_indices = torch.argmax(scores_softmax, dim=2)  # (1,b)
            for i, char in enumerate(current_char_indices.squeeze(0)):
                output_word[i] += self.target_vocab.id2char[char.item()]

        dec_words = []
        for word in output_word:
            ind = word.find(self.target_vocab.id2char[END])
            dec_words.append(word if ind == -1 else word[:ind])

        return dec_words

        ### END YOUR CODE

