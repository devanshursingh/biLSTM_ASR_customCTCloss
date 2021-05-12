import torch
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ASRModelDiscrete(torch.nn.Module):
    """
    Creates a torch LSTM for training a speech recognizer using discrete inputs.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size=257, alphabet_size=28):
        """
        embedding_dim: Dimensions to set embeddings to.
        hidden_dim: Dimensions to set LSTM output to.
        vocab_size: Number of distinct input quantized labels, plus padding.
        alphabet_size: Number of distinct output labels in label sequence, plus blank.
        """
        super(ASRModelDiscrete, self).__init__()
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(embedding_dim, int(hidden_dim/2), batch_first=True, bidirectional=True)#, num_layers=2, dropout=0.4)

        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x, x_lens):
        """
        This architecture consists of two MLP layers before and after a biLSTM. It is simple and has sufficient
        computational power to model the task well.

        x_lens: Tuple with length of each sequence in batch, unpadded. Allows pack_padded_sequence on LSTM inputs.
        """
        embeds = self.word_embeddings(x)

        out = self.relu(self.lin1(embeds))

        out = torch.nn.utils.rnn.pack_padded_sequence(out, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(out)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)

        return log_probs

class ASRModelMFCC(torch.nn.Module):
    """
    Creates a torch LSTM for training a speech recognizer using MFCC continuous vector inputs.
    """
    def __init__(self, mfcc_dim, hidden_dim, alphabet_size=28):
        """
        mfcc_dim: Dimensions of input audio's MFCC frame vector sequence.
        hidden_dim: Dimensions to set combined biLSTM output to.
        alphabet_size: Number of distinct output labels in label sequence, plus blank.
        """
        super(ASRModelMFCC, self).__init__()
        self.mfcc_dim = mfcc_dim

        self.conv1 = nn.Conv1d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv1d(10, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(mfcc_dim, int(hidden_dim/2), batch_first=True, bidirectional=True)#, num_layers=2, dropout=0.4)
        self.decoder = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x, x_lens):
        """
        This architecture consists largely of two convolutional layers on each MFCC vector combined with a
        biLSTM on top, with a final MLP classification layer on top. Regularization schemes include
        droput between the convolutional layers.

        x_lens: Tuple with length of each sequence in batch, unpadded. Allows pack_padded_sequence on LSTM inputs.
        """

        batch_size = x.size()[0]
        seq_length = x.size()[1]
        
        x = x.view(batch_size*seq_length, 1, self.mfcc_dim)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.view(batch_size, seq_length, self.mfcc_dim)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        letter_probs = self.decoder(lstm_out)
        log_probs = F.log_softmax(letter_probs, dim=2)

        return log_probs
