import pdb

import torch
#import librosa
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

        self.dataset = self.load_quantized_features(file_lbls, text=text)

    def load_quantized_features(self, file_lbls, text=None):
        # TODO: might have to use self.phones somehow
        dataset = {}

        # Extract labels for each utterance
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

            for idx in range(len(lbl_seqs)):
                dataset.update({idx: {'feats':lbl_seqs[idx], 'target_tokens':words[idx]}})
        else:
            for idx in range(len(lbl_seqs)):
                dataset.update({idx: {'feats':lbl_seqs[idx]}})

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print('getitem: ', idx)
        if self.text is None:
            return self.dataset[idx]['feats'], None
        #print(self.dataset[idx]['feats'])
        #print(self.dataset[idx]['target_tokens'])
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
        phones = {'<sil>':27,
                    'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7,
                    'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16,
                    'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22,
                    'w':23, 'x':24, 'y':25, 'z':26}
        phones_rev = {v: k for k, v in phones.items()}
        self.phones = phones
        self.phones_rev = phones_rev
        self.text = text
        self.dataset = self.compute_mfcc(wav_scp, wav_dir, text=text)

    def compute_mfcc(self, wav_scp, wav_dir, text=None):
        dataset = {}
        # TODO: Create a dataset with mfcc features
        # filename='data/
        #y, sr = librosa.load()
        #librosa.feature.mfcc(y=y, sr=sr)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.text is None:
            return self.dataset[idx]['feats']
        return self.dataset[idx]['feats'], self.dataset[idx]['target_tokens']
