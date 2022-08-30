import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import ipdb
from config import args, device
import torch.nn.functional as F
import random
from config import args

class REModel(nn.Module):
    def __init__(self, hidden_size=768):
        super(REModel, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(args.model)
        tokenizer.add_tokens(["[s1]", "[s2]"])
        self.bert = BertModel.from_pretrained(args.model)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.proj_relation = nn.Linear(2 * hidden_size, 37)
        self.proj_trigger = nn.Linear(hidden_size, 2)
        self.proj_binary = nn.Linear(hidden_size, 2)

    def reset(self, class_cnt, hidden_size=768):
        self.proj_relation = nn.Linear(2 * hidden_size, class_cnt)



    def forward(self, inputs):
        # Phase 1: feed inputs to self.bert and extract the hidden states
        # Phase 2-1: keep the context part unmasked, and apply start & end prediction
        # Phase 2-2: concatenate hid_trigger, x, and y,
        # then pass this concatenated tensor to self.proj_relation
        last_hidden_states = self.bert(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])["last_hidden_state"]

        start_end_logit = self.proj_trigger(last_hidden_states) # shape: (batch_size, seq_len, 2)

        ids = []
        for x_idx in inputs['x_idx']:
            ids.append([0, x_idx[0]-1])
        masked_start_end_logit = self.get_masked(
            start_end_logit,
            torch.tensor(ids),
            mask_val=float('-inf')
        )

        ids = self.get_triggers_ids(masked_start_end_logit)

        trigger = self.attention(last_hidden_states, torch.tensor(ids))

        x = []
        for b_idx in range(len(last_hidden_states)):
            x.append(last_hidden_states[b_idx][inputs['x_idx'][b_idx][1]+1, :])
        x = torch.vstack(x)


        concat_hid = torch.hstack((trigger, last_hidden_states[:, 0, :]))
        relation_logit = self.proj_relation(concat_hid)
        binary_logit = self.proj_binary(x)

        return relation_logit, start_end_logit, binary_logit#, distributions#, tri_len_logit

    def get_triggers_ids(self, masked_start_end_logit, tri_len=None):
        ids = []

        if tri_len is None:
            for batch_idx, sample in enumerate(masked_start_end_logit):
                start = sample[:,0] # start: shape (512)
                end = sample[:,1]
                start_candidates = torch.topk(start, k=30)
                end_candidates = torch.topk(end, k=30)
                ans_candidates = [(0, 1)]
                scores = [-100]
                start_logits = F.softmax(start_candidates[0])
                end_logits = F.softmax(end_candidates[0])
                for i, s in enumerate(start_candidates[1]):
                    for j, e in enumerate(end_candidates[1]):
                        if s == 0:
                            ans_candidates.append((s, s+1))
                            # scores.append(start_candidates[0][i]+end_candidates[0][j])
                            scores.append(start_logits[i] * end_logits[j])
                        if s<e and e-s <= 10:
                            ans_candidates.append((s, e))
                            # scores.append(start_candidates[0][i]+end_candidates[0][j])
                            scores.append(start_logits[i] * end_logits[j])
                results = list(zip(scores, ans_candidates))
                results.sort()
                results.reverse()

                ids.append([int(results[0][1][0]), int(results[0][1][1])])
            return ids
        else:
            for batch_idx, sample in enumerate(masked_start_end_logit):
                start = sample[:,0] # start: shape (512)
                end = sample[:,1]
                start_logits = F.softmax(start)
                end_logits = F.softmax(end)
                # start_candidates = torch.topk(start, k=30)
                # end_candidates = torch.topk(end, k=30)
                max_score = float('-inf')
                cand = None
                for i in range(len(start_logits)-tri_len[batch_idx]):
                    # for j in range(i+1, len(end_logits)):
                    #     if j - i <= 14:
                    cur_score = start_logits[i] + end_logits[i+tri_len[batch_idx]]
                    if cur_score > max_score:
                        max_score = cur_score
                        cand = [i, i+tri_len[batch_idx]]
                ids.append(cand)
            return ids



    def infer(self, inputs):
        last_hidden_states = self.bert(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])["last_hidden_state"]
        start_end_logit = self.proj_trigger(last_hidden_states) # shape: (batch_size, seq_len, 2)

        ids = []
        for x_idx in inputs['x_idx']:
            ids.append([0, x_idx[0]-1])
        masked_start_end_logit = self.get_masked(
            start_end_logit,
            torch.tensor(ids),
            mask_val=float('-inf')
        )

        # y = []
        # for b_idx in range(len(last_hidden_states)):
        #     y.append(last_hidden_states[b_idx][inputs['x_idx'][b_idx][0]-1, :])
        # y = torch.vstack(y)
        # tri_len_logit = self.proj_tri_len(y)
        # tri_len = torch.argmax(tri_len_logit, dim=1)

        # ids = self.get_triggers_ids(masked_start_end_logit, tri_len)
        ids = self.get_triggers_ids(masked_start_end_logit)
        tokenizer = BertTokenizer.from_pretrained(args.model)
        tokenizer.add_tokens(["[s1]", "[s2]"])
        p_trigs, gt_trigs = [], []
        for i in range(len(inputs['input_ids'])):
            p_trig = tokenizer.decode(inputs['input_ids'][i][ids[i][0] : ids[i][1]])
            p_trigs.append(p_trig)
            gt_trig = tokenizer.decode(inputs['input_ids'][i]\
                                 [inputs['t_idx'][i][0] : inputs['t_idx'][i][1]])
            gt_trigs.append(gt_trig)


        # trigger = self.attention(last_hidden_states, inputs['t_idx'])
        trigger = self.attention(last_hidden_states, torch.tensor(ids))
        # trigger, inv_lengths = self.get_trigger_and_lengths(last_hidden_states, torch.tensor(ids))
        # trigger, inv_lengths = self.get_trigger_and_lengths(last_hidden_states,  inputs['t_idx'])
        # trigger = torch.mul(trigger.sum(dim=1), inv_lengths.to(device))
        # if len(trigger.shape) == 2:
        #     trigger = trigger.unsqueeze(dim=1)
        # cls_and_trig = torch.hstack((last_hidden_states[:, 0, :].unsqueeze(dim=1), trigger))
        # concat_hid = torch.mul(
        #     cls_and_trig.sum(dim=1),
        #     (1 / (1 + (1 / inv_lengths))).to(device)
        # )

        # trigger = self.get_masked(last_hidden_states, torch.tensor(ids)).mean(dim=1)
        # trigger = self.get_masked(last_hidden_states, inputs['t_idx']).mean(dim=1)
        # _, trigger = self.rnn(
        #     self.get_trigger(last_hidden_states, torch.tensor(ids))
        # )
        # trigger = trigger.view(trigger.shape[1], -1)
        # _, trigger = self.rnn(
        #     self.get_trigger(last_hidden_states, inputs['t_idx'])
        # )
        # trigger = trigger.view(trigger.shape[1], -1)

        # x = self.get_masked(last_hidden_states, inputs['x_idx']).mean(dim=1)
        # y = self.get_masked(last_hidden_states, inputs['y_idx']).mean(dim=1)
        # concat_hid = torch.hstack((trigger, x, y))
        # trigger = self.get_masked(last_hidden_states, torch.tensor(ids)).mean(dim=1)
        # no_trig = [i for i in range(len(ids)) if ids[i][0] == 0]
        # no_trig = [i for i in range(len(inputs['t_idx'])) if inputs['t_idx'][i][0] == 0]
        # for i in no_trig:
        #     trigger[i, :] = self.uni_trigger[:]
        x = []
        for b_idx in range(len(last_hidden_states)):
            x.append(last_hidden_states[b_idx][inputs['x_idx'][b_idx][1]+1, :])
        x = torch.vstack(x)
        # concat_hid = torch.hstack((trigger, x))
        binary_logit = self.proj_binary(x)
        bin_pred = torch.argmax(binary_logit, dim=1)

        # cls = []
        for batch_idx in range(len(bin_pred)):
            if bin_pred[batch_idx] == 0:
                # trigger[batch_idx][:] = last_hidden_states[batch_idx, 0, :]
                # cls.append(last_hidden_states[batch_idx, 0, :])
            # if inputs['has_trigger'][batch_idx] == 0:
                trigger[batch_idx][:] = torch.zeros(len(last_hidden_states[batch_idx, 0, :]))
            # else:
            #     cls.append(torch.zeros(len(last_hidden_states[batch_idx, 0, :]), device=device))
        # cls = torch.vstack(cls)


        # For bert baseline (only use bert and no other modules)
        #for batch_idx in range(len(inputs['has_trigger'])):
        #    trigger[batch_idx][:] = torch.zeros(len(last_hidden_states[batch_idx, 0, :]))

        concat_hid = torch.hstack((trigger, last_hidden_states[:, 0, :]))
        # concat_hid = torch.hstack((trigger, cls))
        # concat_hid = (trigger + cls) / 2
        # concat_hid = (trigger + last_hidden_states[:, 0, :]) / 2
        # relation_logit = self.proj_relation(
        #     torch.sigmoid(self.proj_relation0(concat_hid))
        # )
        relation_logit = self.proj_relation(concat_hid)
        # return torch.argmax(relation_logit, dim=1), p_trigs, gt_trigs
        argmax = torch.argmax(relation_logit, dim=1)
        has_trigger = torch.argmax(binary_logit, dim=1)

        return argmax, has_trigger, ids, p_trigs, gt_trigs


    def get_masked(self, mat, ids, mask_val=0):
        batch_size, seq_len, cls = mat.shape
        mask = torch.ones(batch_size, seq_len, cls)
        for i in range(batch_size):
            mask[i, ids[i][0]:ids[i][1], :] = 0
        mask = mask.bool()
        return mat.masked_fill(mask.to(device), mask_val)

    def get_trigger(self, mat, ids, mask_val=0, length=15):
        batch_size, seq_len, cls = mat.shape
        triggers = []
        for b_id in range(batch_size):
            # self.total += 1
            trigger = mat[b_id][ids[b_id][0] : ids[b_id][1]][:]
            if len(trigger) < length:
                padding = torch.zeros(length - len(trigger), cls)
                padding = padding.to(device)
                trigger = torch.vstack((trigger, padding))
            else:
                # self.long_trig += 1
                trigger = trigger[:length]
            try:
                assert len(trigger) == length
            except:
                ipdb.set_trace()
            triggers.append(trigger)

        return torch.vstack(triggers).view(batch_size, -1, cls)
    
    def attention(self, mat, ids):
        triggers = []
        batch_size, _, _ = mat.shape
        for b_id in range(batch_size):
            trigger = mat[b_id][ids[b_id][0] : ids[b_id][1]][:]
            score = []
            cls = mat[b_id, 0, :]
            # score.append(torch.dot(cls, cls))
            for j in range(len(trigger)):
                score.append(torch.dot(cls, trigger[j]))
            score = torch.tensor(score, device=device)
            score = F.softmax(score)
            # trigger = torch.vstack((cls, trigger))
            triggers.append(torch.matmul(trigger.T, score))
        return torch.vstack(triggers)


    def get_trigger_and_lengths(self, mat, ids, mask_val=0):
        lengths = []
        batch_size, seq_len, cls = mat.shape
        mask = torch.ones(batch_size, seq_len, cls)
        for i in range(batch_size):
            mask[i, ids[i][0]:ids[i][1], :] = 0
        mask = mask.bool()
        for b_id in range(batch_size):
            # self.total += 1
            lengths.append(ids[b_id][1] - ids[b_id][0])
        return mat.masked_fill(mask.to(device), mask_val), \
               1 / torch.vstack(lengths)
