import torch
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASRModelDiscrete(torch.nn.Module):
    def __init__(self, vocab_size=257, alphabet_size=27, embedding_dim=100, hidden_dim=50):
        super(ASRModelDiscrete, self).__init__()
        # TODO: Initialize Model
        self.hidden_dim = hidden_dim
        # NOTE: vocab size is 256, but one extra for padding_idx
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x):
        # TODO: Write Forward Pass
        embeds = self.word_embeddings(x)
        #print(embeds.view(len(x), 1, -1).size())
        lstm_out, _ = self.lstm(embeds)
        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)
        # Output should be Batch × InputLength × NumLetters
        return log_probs

class ASRModelMFCC(torch.nn.Module):
    def __init__(self):
        super(ASRModelMFCC, self).__init__()
        # TODO: Initialize Model

    def forward(self, x):
        output = x
        # TODO: Write Forward Pass
        return output
