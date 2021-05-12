import pdb

import argparse
import numpy as np
import torch

from dataset import Project03DatasetMFCC, Project03DatasetDiscrete
from model import ASRModelMFCC, ASRModelDiscrete
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from ctc_loss import CTCLoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.special import softmax

parser = argparse.ArgumentParser(description='This is the code for performing recognition with a 50 word vocabulary')
parser.add_argument('--stage', type=int, default=0, help='What stage of training to start')
parser.add_argument('--features', type=str, default='Discrete', choices=['Discrete','MFCC'], help='Use discrete features or mfcc features')
parser.add_argument('--max-epochs', type=int, default=100, help='number of epochs to run')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--custom-ctc', action="store_true", help='use custom ctc loss function')
args = parser.parse_args()

def collate_fn(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    data_lens = [len(x) for x in data]
    target_lens = [len(x) for x in targets]
    data_pad = pad_sequence(data, batch_first=True, padding_value=0)
    target_pad = pad_sequence(targets, batch_first=True, padding_value=0)
    return {'feats':data_pad, 'targets':target_pad, 'feat_lens':data_lens, 'target_lens':target_lens}


def get_vocab(file_name, func=lambda x: x, skip_header=True):
    print("reading file: %s" % file_name)
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()  # skip the header
        for line in fin:
            if len(line.strip()) == 0:
                continue
            fields = func(line.strip())
            res.append(fields)
    print("%d lines, done" % len(res))
    res = list(set(res))
    return res


def train(train_dataloader, val_dataloader, model, criterion, optim):
    model.train()
    total_loss = 0.
    num_batches = 0
    for i, item in enumerate(train_dataloader):
        input_seq = item['feats']
        target_seq = item['targets']
        input_lens = item['feat_lens']
        target_lens = item['target_lens']
        model.zero_grad()
        pred_seq = model(input_seq, input_lens)
        # need to swap batch and seq length dims to fit CTC loss fn requirements
        loss = criterion(torch.swapdims(pred_seq, 0, 1), target_seq, input_lens, target_lens)
        total_loss+=loss.item()

        loss.backward()
        optim.step()

        num_batches+=1

    total_loss/=num_batches
    

    model.eval()
    val_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for i, item in enumerate(val_dataloader):
            input_seq_val = item['feats']
            target_seq_val = item['targets']
            input_lens_val = item['feat_lens']
            target_lens_val = item['target_lens']

            pred_seq_val = model(input_seq_val, input_lens_val)

            loss = criterion(torch.swapdims(pred_seq_val, 0, 1), target_seq_val, input_lens_val, target_lens_val)
            val_loss+=loss.item()
            num_batches+=1

    val_loss/=num_batches

    return total_loss, val_loss


def test(test_dataset, phones, phones_rev, vocab, model, criterion, w_labels=True):
    """
    The following are added variables. The rest are self-explanatory.

    phones, phones_rev: objects from custom Dataset class, allows testing on val_dataset, which
    is not a Dataset object.
    w_labels: indicates whether the input dataset has labels or not, so that accuracy can be calculated.
    """
    
    model.eval()
    utt_correct = 0
    utt_total = 0
    hyp_string = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            input_seq, target_seq = test_dataset[i]

            # add batch dim
            input_seq = torch.unsqueeze(input_seq, dim=0)
            
            if w_labels:
                # turn into string
                target_word = ''
                for widx in target_seq.tolist():
                    target_word+=phones_rev.get(widx)

            input_len = [int(input_seq.size()[1])]
            pred_seq = model(input_seq, input_len)

            losses = []
            for word in vocab:
                # Turn word into a tensor of letter indices
                w_tensor = []
                word = '_' + word + '_'
                for let in list(word):
                    w_idx = phones.get(let)
                    w_tensor.append(w_idx)
                target = torch.unsqueeze(torch.tensor(w_tensor), dim=0)

                input_len = (pred_seq.size()[1],)
                target_len = (target.size()[1],)

                loss = criterion(torch.swapdims(pred_seq, 0, 1), target, input_len, target_len)
                losses.append(loss.item())

            losses = -1. * np.array(losses)
            pred_word = vocab[np.argmax(losses)]
            confidence = np.amax(softmax(losses))
            hyp_string.append((pred_word, confidence))
            # print("pred: _", pred_word, "_")
            if w_labels:
                # print("target: ", target_word)
                # print(" ")
                pred_word = '_' + pred_word + '_'
                if target_word == pred_word:
                    utt_correct+=1
            utt_total+=1
    
    return hyp_string, utt_correct / utt_total


if __name__ == "__main__":
    print('Stage 1: Preparing Dataset and DataLoader')
    vocab = get_vocab("data/clsp.trnscr")
    model = None
    if args.custom_ctc:
       criterion = CTCLoss()
    else:
       criterion = torch.nn.CTCLoss()
    if args.features == 'Discrete':
        train_dataset = Project03DatasetDiscrete('data/clsp.trnlbls', text='data/clsp.trnscr')
        train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2, shuffle=True, stratify=train_dataset.word_labels)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        val_dataset = []
        for vid in val_idx.tolist():
            val_dataset.append(train_dataset[vid])
        train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, sampler=val_sampler)
        test_dataset = Project03DatasetDiscrete('data/clsp.devlbls')
        model = ASRModelDiscrete(embedding_dim=300, hidden_dim=200)
    else:
        train_dataset = Project03DatasetMFCC('data/clsp.trnwav', 'data/waveforms', text='data/clsp.trnscr')
        train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2, shuffle=True, stratify=train_dataset.word_labels)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        val_dataset = []
        for vid in val_idx.tolist():
            val_dataset.append(train_dataset[vid])
        train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, sampler=val_sampler)
        test_dataset = Project03DatasetMFCC('data/clsp.trnwav', 'data/waveforms')
        model = ASRModelMFCC(mfcc_dim=100, hidden_dim=200)

    if args.stage <= 1:
        print('Stage 2: Training ' + args.features + ' Model')
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []
        val_losses = []
        for epoch in range(args.max_epochs):
            train_loss, val_loss = train(train_dataloader, val_dataloader, model, criterion, optim)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('\tepoch ' + str(epoch) + ': train_loss=' + str(train_loss) + '  |  val_loss=' + str(val_loss))
            torch.save(model.state_dict(), 'checkpoint/model_epoch'+str(epoch)+'.pt')
            torch.save(model.state_dict(), 'checkpoint/model_last.pt')

        # Plot train and val loss
        fig0=plt.figure(0)
        plt.plot(train_losses, color='red', label='train')
        plt.xlabel('Epoch number')
        plt.ylabel('Epoch loss')
        plt.plot(val_losses, color='blue', label='val')
        plt.legend()
        plt.show()

    if args.stage <= 2:
        print('Stage 3: Test ' + args.features + ' Model')
        model.load_state_dict(torch.load('checkpoint/model_last.pt'))
        predictions, accuracy = test(train_dataset, train_dataset.phones, train_dataset.phones_rev, vocab, model, criterion)
        print('Final Train Accuracy: ' + str(accuracy))
        predictions, accuracy = test(val_dataset, train_dataset.phones, train_dataset.phones_rev, vocab, model, criterion)
        print('Final Validation Accuracy: ' + str(accuracy))
        predictions, accuracy = test(test_dataset, train_dataset.phones, train_dataset.phones_rev, vocab, model, criterion, w_labels=False)
        print('Final Test Accuracy: ' + str(accuracy))

        out_file = open(f"{args.features}_test_results.txt", "w")
        out_file.write("test_results.txt\n")
        out_file.write("predicted_word\tconfidence\n")
        for most_likely_word, confidence in predictions:
            out_file.write(f'{most_likely_word}\t{confidence}\n')
        out_file.close()
