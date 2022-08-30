from model import REModel
import numpy as np
from pathlib import Path
from dataset import REDataset
import argparse
import pickle
from config import args, device
import json
import sys
import torch
import torch.nn as nn
import ipdb
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.optim as optim
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score
from cal import cal_seen_unseen_stats

def load_model(path: Path) -> REModel:
    model = REModel()
    #model.reset(13)
    model.to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    del ckpt
    return model

def test(model=None, epoch=0):
    test_set = REDataset(pickle.load(open(args.test_path, 'rb')))
    
    if model is None:
        model = load_model(args.load_path)
    #model.reset(13)
    model.to(device)

    test_loader = DataLoader(
        test_set,
        args.batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn
    )
    y_true, y_pred = [], []
    model.eval()

    preds, gts = [], []
    bin_pred, bin_gt = [], []

    has_trig_preds, has_trig_gt = [], []
    no_trig_preds, no_trig_gt = [], []


    trigger_length = {}
    start_pred, end_pred = [], []
    start_gt, end_gt = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            y_true.extend([label.item() for label in batch['label']])
            bin_gt.extend([label.item() for label in batch['has_trigger']])
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            pred, has_trigger, ids, p_trigs, gt_trigs  = model.infer(batch)

            for i in range(len(batch['label'])):
                cur_trig_len = (ids[i][1] - ids[i][0])
                if not type(cur_trig_len) == int: cur_trig_len = cur_trig_len.item()
                trigger_length[cur_trig_len] = trigger_length.get(cur_trig_len, 0) + 1
                start_pred.append(ids[i][0])
                end_pred.append(ids[i][1])
                start_gt.append(batch['t_idx'][i][0])
                end_gt.append(batch['t_idx'][i][1])


            preds.extend(p_trigs)
            gts.extend(gt_trigs)

            y_pred.extend([label.item() for label in pred])
            bin_pred.extend([label.item() for label in has_trigger])
            for i in range(len(batch["label"])):
                if batch["has_trigger"][i] == 1:
                    has_trig_gt.append(batch['label'][i].item())
                    has_trig_preds.append(pred[i].item())
                else:
                    no_trig_gt.append(batch['label'][i].item())
                    no_trig_preds.append(pred[i].item())

        cal_metric(preds, gts)
        print(f1_score(y_true, y_pred, average='macro'))
        all_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"all: {all_f1}")
        no_trigger_f1 = f1_score(no_trig_gt, no_trig_preds, average='micro')
        print(f"no trig: {no_trigger_f1}")
        has_trigger_f1 = f1_score(has_trig_gt, has_trig_preds, average='micro')
        print(f"has trig: {has_trigger_f1}")
        bin_acc = f1_score(bin_gt, bin_pred, average='micro')
        print(f"bin_acc: {bin_acc}")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print((y_pred==y_true).sum() / len(np.array(y_true)))


        with open(f"{epoch}_diff.csv", 'w') as f:
            diff = list(zip(gts, preds))
            diff = [','.join([pair[0], pair[1]]) for pair in diff]
            print("\n".join(diff), file=f)

        with open(f"{epoch}_preds.txt", 'w') as f1:
            print("\n".join(preds), file=f1)
        with open(f"{epoch}_gts.txt", 'w') as f2:
            print("\n".join(gts), file=f2)
        np.savetxt(f'{epoch}_y_pred.txt', np.array(y_pred), fmt='%d')
        np.savetxt(f'{epoch}_y_true.txt', np.array(y_true), fmt='%d')

        print("(un)seen stats:")
        cal_seen_unseen_stats(y_pred, y_true)
        return all_f1

def cal_metric(preds, gts):
    assert len(preds) == len(gts)
    false_pos = 0
    false_neg = 0
    exact_match = 0
    cls_cnt = 0
    f = open('diff.csv', 'w')
    res = []
    for i in range(len(preds)):
        res.append(preds[i] + "," + gts[i])
        if gts[i] == '[CLS]':
            cls_cnt += 1
        if gts[i] == '[CLS]' and preds[i] != '[CLS]':
            false_pos += 1
        if gts[i] != '[CLS]' and preds[i] == '[CLS]':
            false_neg += 1
        if gts[i] != '[CLS]' and preds[i] == gts[i]:
            exact_match += 1
    print(f"fp: {false_pos}")
    print(f"fn: {false_neg}")
    print(f"em: {exact_match}")
    print(f"cls_cnt: {cls_cnt}")
    print(f"total amount: {len(preds)}")
    print("\n".join(res), file=f)

if __name__ == '__main__':
    test()
