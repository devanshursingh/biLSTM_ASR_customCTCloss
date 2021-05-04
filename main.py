import pdb

import argparse
import numpy as np
import torch

from dataset import Project03DatasetMFCC, Project03DatasetDiscrete
from model import ASRModelMFCC, ASRModelDiscrete
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='This is the code for performing recognition with a 50 word vocabulary')
parser.add_argument('--stage', type=int, default=0, help='What stage of training to start')
parser.add_argument('--features', type=str, default='Discrete', choices=['Discrete','MFCC'], help='Use discrete features or mfcc features')
parser.add_argument('--max-epochs', type=int, default=100, help='number of epochs to run')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
args = parser.parse_args()

def collate_fn(batch):
    data = [item[0] for item in batch]
    #print(data)
    targets = [item[1] for item in batch]
    #print(targets)
    data_lens = [len(x) for x in data]
    #print(data_lens)
    target_lens = [len(x) for x in targets]
    #print(target_lens)
    data_pad = pad_sequence(data, batch_first=True, padding_value=0)
    #print(data_pad)
    target_pad = pad_sequence(targets, batch_first=True, padding_value=0)
    #print(target_pad)
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
    return res


def train(train_dataloader, model, criterion, optim):
    # TODO: Fill in the training
    model.train()
    total_loss = 0
    num_batches = 0
    for i, item in enumerate(train_dataloader):
        print(i)
        input_seq = item['feats']
        target_seq = item['targets']
        input_lens = item['feat_lens']
        target_lens = item['target_lens']
        model.zero_grad()
        # what would happen to the pad value 0, gets multiplied to weights and makes them 0
        #input_seq_unsqueezed = torch.unsqueeze(input_seq, dim=1)
        pred_seq = model(input_seq)
        loss = criterion(torch.swapdims(pred_seq, 0, 1), target_seq, input_lens, target_lens)
        total_loss+=loss.item()

        loss.backward()
        optim.step()

        num_batches+=1
    
    model.eval()

    # TODO: implement validation
    # val_loss = 0.
    # with torch.no_grad():
    #     for batch, (context, target) in enumerate(val_dataloader):
    #         prediction = model(context)
    #         loss = loss_fn(prediction, target)
    #         test_loss += loss

    # test_loss/=len(test_data)

    return total_loss / num_batches


def test(test_dataset, vocab, model):
    model.eval()
    utt_correct = 0
    utt_total = 0
    with torch.no_grad():
        for i in range(len(test_dataset)):
            print(i)
            # TODO: Fill in this part to choose the best word for each utterance in the test data
            pred = model(test_dataset[i]['feat'])
            losses = []
            input_length = test_dataset[i]['feat_lens']
            target_length = test_dataset[i]['target_lens']
            for word in vocab:
                # Turn word into a tensor of letter indices
                w_tensor = []
                for let in list(word):
                    w_idx = test_dataset.phones.get(let)
                    w_tensor.append(w_idx)
                target = torch.tensor(w_tensor)

                loss = criterion(pred, target, input_length, target_length)
                losses.append(loss)
            pred_word = vocab[np.argmin(np.array(losses))]
        
    return hyp_string, utt_correct / utt_total


if __name__ == "__main__":
    print('Stage 1: Preparing Dataset and DataLoader')
    vocab = get_vocab("data/clsp.trnscr")
    model = None
    if args.features == 'Discrete':
        train_dataloader = DataLoader(Project03DatasetDiscrete('data/clsp.trnlbls', text='data/clsp.trnscr'), batch_size=128, collate_fn=collate_fn)
        train_dataset = Project03DatasetDiscrete('data/clsp.trnlbls', text='data/clsp.trnscr')
        test_dataset = Project03DatasetDiscrete('data/clsp.devlbls')
        model = ASRModelDiscrete()
    else:
        train_dataloader = DataLoader(Project03DatasetMFCC('data/clsp.trnwav', 'data/waveforms', text='data/clsp.trnscr'), batch_size=128, collate_fn=collate_fn)
        train_dataset = Project03DatasetMFCC('data/clsp.trnwav', 'data/waveforms', text='data/clsp.trnscr')
        test_dataset = Project03DatasetMFCC('data/clsp.trnwav', 'data/waveforms')
        model = ASRModelMFCC()

    if args.stage <= 1:
        print('Stage 2: Training ' + args.features + ' Model')
        criterion = torch.nn.CTCLoss()
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.max_epochs):
            train_loss = train(train_dataloader, model, criterion, optim)
            print('\tepoch ' + str(epoch) + ': train_loss=' + str(train_loss))
            torch.save(model.state_dict(), 'checkpoint/model_epoch'+str(epoch)+'.pt')
            torch.save(model.state_dict(), 'checkpoint/model_last.pt')

    if args.stage <= 2:
        print('Stage 3: Test ' + args.features + ' Model')
        model.load_state_dict(torch.load('checkpoint/model_last.pt'))
        predictions, accuracy = test(train_dataset, vocab, model)
        print('Final Train Accuracy: ' + str(accuracy))
        predictions, accuracy = test(test_dataset, vocab, model)
        print('Final Test Accuracy: ' + str(accuracy))
