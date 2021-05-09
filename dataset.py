import pdb

import torch
import librosa
import os
import numpy as np

from torch.utils.data import Dataset


class Project03DatasetDiscrete(Dataset):
    """Dataset for Information Extraction Project 03"""

    def __init__(self, file_lbls, lbl_names='data/clsp.lblnames', text=None):
        """
        Args:
            file_lbls (string): the clsp.trnlbls file
            lbl_names (string): the clsp.lblnames file
            wav_scp (string): the clsp.trnwav file
            wav_dir (string): the directory where the .wav files are
            text (string optional): the text transcriptions of each
        """
        phones = {'_':27,
                'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7,
                'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16,
                'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22,
                'w':23, 'x':24, 'y':25, 'z':26}
        phones_rev = {v: k for k, v in phones.items()}
        self.phones = phones
        self.phones_rev = phones_rev
        self.text = text

        # Create vocab and label to index
        lblnames = []
        with open(lbl_names, 'r') as n:
            lines = n.readlines()[1:]
            for l in lines:
                lblnames.append(l.strip('\n'))
        vocab_index = [num for num in range(1,len(lblnames)+1)]
        self.vocab = {lblnames[i] : vocab_index[i] for i in range(len(lblnames))}

        # Also create word_labels, used in train_test_split
        self.word_labels = []
        self.dataset = self.load_quantized_features(file_lbls, text=text)
        self.word_labels = np.array(self.word_labels)

    def load_quantized_features(self, file_lbls, text=None):
        dataset = {}

        # Extract labels for each utterance and convert to index tensor
        lbl_seqs = []
        with open(file_lbls, 'r') as t:
            lines = t.readlines()[1:]
            for j in range(len(lines)):
                lbls = lines[j].split(" ")[:-1]
                l_tensor = []
                for lbl in lbls:
                    l_idx = self.vocab.get(lbl)
                    l_tensor.append(l_idx)
                lbl_seqs.append(torch.tensor(l_tensor))

        # Read in the words and convert to index tensor
        if text is not None:
            words = []
            with open(text, 'r') as s:
                lines = s.readlines()[1:]
                for l in lines:
                    w = '_' + l.strip('\n') + '_'
                    w_tensor = []
                    for let in list(w):
                        w_idx = self.phones.get(let)
                        w_tensor.append(w_idx)
                    words.append(torch.tensor(w_tensor))

            for idx in range(len(lbl_seqs)):
                dataset.update({idx: {'feats':lbl_seqs[idx], 'target_tokens':words[idx]}})
                # read in unique label for each utterance, used in train_test_split
                w_lbl = int(''.join(map(str, words[idx].detach().numpy().tolist())))
                self.word_labels.append(w_lbl)
        else: # don't read in words if creating test dataset
            for idx in range(len(lbl_seqs)):
                dataset.update({idx: {'feats':lbl_seqs[idx]}})

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.text is None:
            return self.dataset[idx]['feats'], None
        return self.dataset[idx]['feats'], self.dataset[idx]['target_tokens']


class Project03DatasetMFCC(Dataset):
    """Dataset for Information Extraction Project 03"""

    def __init__(self, wav_scp, wav_dir, text=None):
        """
        Args:
            wav_scp (string): the clsp.trnwav file
            wav_dir (string): the directory where the .wav files are
            text (string optional): the text transcriptions of each
        """
        phones = {'_':27,
                    'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7,
                    'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16,
                    'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22,
                    'w':23, 'x':24, 'y':25, 'z':26}
        phones_rev = {v: k for k, v in phones.items()}
        self.phones = phones
        self.phones_rev = phones_rev
        self.text = text
        # Also create word_labels, used in train_test_split
        self.word_labels = []
        self.dataset = self.compute_mfcc(wav_scp, wav_dir, text=text)
        self.word_labels = np.array(self.word_labels)

    def compute_mfcc(self, wav_scp, wav_dir, text=None):
        dataset = {}

        # Extract mfcc features for each utterance
        mfccs = []
        with open(wav_scp, 'r') as t:
            paths = t.readlines()[1:]
            for j in range(len(paths)):
                path = wav_dir + '/' + paths[j].strip('\n')
                audio, sr = librosa.load(path, sr=None)
                # NOTE hyperparameters: 40 dim vectors, 25 ms window length, 10 ms stride
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=int(0.025*sr), hop_length=int(0.010*sr))
                mfccs.append(torch.swapdims(torch.tensor(mfcc), 0, 1))

        # Read in the words
        if text is not None:
            words = []
            with open(text, 'r') as s:
                lines = s.readlines()[1:]
                for l in lines:
                    w = '_' + l.strip('\n') + '_'
                    w_tensor = []
                    for let in list(w):
                        w_idx = self.phones.get(let)
                        w_tensor.append(w_idx)
                    words.append(torch.tensor(w_tensor))

            for idx in range(len(mfccs)):
                dataset.update({idx: {'feats':mfccs[idx], 'target_tokens':words[idx]}})
                w_lbl = int(''.join(map(str, words[idx].detach().numpy().tolist())))
                self.word_labels.append(w_lbl)
        else:
            for idx in range(len(mfccs)):
                dataset.update({idx: {'feats':mfccs[idx]}})

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.text is None:
            return self.dataset[idx]['feats'], None
        return self.dataset[idx]['feats'], self.dataset[idx]['target_tokens']
