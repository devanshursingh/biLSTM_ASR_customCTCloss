import torch
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ASRModelDiscrete(torch.nn.Module):
    """
    Creates a torch LSTM for training a speech recognizer using discrete inputs.

    vocab_size: Number of distinct input quantized labels, plus padding.
    alphabet_size: Number of distinct output labels in label sequence, plus blank.
    embedding_dim: Dimensions to set embeddings to.
    hidden_dim: dimensions to set LSTM output to.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, vocab_size=257, alphabet_size=28):
        super(ASRModelDiscrete, self).__init__()
        self.hidden_dim = hidden_dim
        # NOTE: vocab size is 256, but one extra for padding_idx
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # TODO: try dropout, batch norm

        #self.conv1 = nn.Conv1d(1,1,3, padding=1)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.bilstm1 = nn.LSTM(embedding_dim, int(hidden_dim/2), batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_dim, int(hidden_dim/2), batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()

        # The linear layer that maps from hidden state space to tag space
        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x, x_lens):
        embeds = self.word_embeddings(x)

        #embeds = self.conv1(embeds)

        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm1(embeds)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.relu(lstm_out)
        
        lstm_out = torch.nn.utils.rnn.pack_padded_sequence(lstm_out, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm2(lstm_out)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)

        return log_probs

class ASRModelMFCC(torch.nn.Module):
    def __init__(self, mfcc_dim=40, hidden_dim=100, alphabet_size=28):
        super(ASRModelMFCC, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(mfcc_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x, x_lens):
        # TODO: Write Forward Pass
        #print(embeds.view(len(x), 1, -1).size())
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)
        # Output should be Batch × InputLength × NumLetters
        return log_probs
