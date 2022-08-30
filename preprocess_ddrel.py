from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import ipdb
import spacy
import ipdb
from contraction1 import contractions_dict
from config import args

import re
contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


MAX_LEN = 512

def lemmatize(text):
    text = nlp(text)
    return " ".join([(t.text) for t in text if t.pos_ not in ['PUNCT', 'SPACE'] or t.lemma_ == ':']).replace(" '", "'").replace(" n't", "n't")

six_cls_map = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:4,10:5, 11:6, 12:6,13:6}
four_cls_map = {1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:3,10:4, 11:4, 12:4,13:4}

def preprocess(PATH: str, mode: str) -> None:
    def is_speaker(a):
        a = a.split()
        return (len(a) == 2 and a[0] == "speaker" and a[1].isdigit())
    def rename(d: str, x:str, y: str) -> (str, str, str):
        unused = ["[s1]", "[s2]"]
        a = []
        if is_speaker(x):
            a += [x]
        else:
            a += [None]
        if x != y and is_speaker(y):
            a += [y]
        else:
            a += [None]
        for i in range(len(a)):
            if a[i] is None:
                continue
            d = d.replace(a[i] + ":", unused[i] + " :")
            if x == a[i]:
                x = unused[i]
            if y == a[i]:
                y = unused[i]
        return d, x, y

    tokenizer = BertTokenizer.from_pretrained(args.model)
    tokenizer.add_tokens(["[s1]", "[s2]"])
    #data = json.load(open(, 'r'))
    f = open(PATH, encoding='utf8')
    data = f.readlines()
    data = [json.loads(line.strip()) for line in data]
    examples = []
    for i in tqdm(range(len(data))):
        d = expand_contractions(' '.join(data[i]['context']))
        d = d.replace("A:", '[s1]:').replace("B:", "[s2]:")
        x = '[s1]'
        y = '[s2]'
        label = data[i]['label']
        t_start = 0
        t_end = 1
        trigger = ''
        text = d + '[SEP]' + x + '[CLS]' + y
        attn_mask = torch.ones(512)
        tmp_ids = tokenizer(text).input_ids[1:-1]
        x_st = len(tokenizer(d).input_ids)
        if len(tmp_ids) < MAX_LEN - 2:
            text += '[PAD] ' * (MAX_LEN - 2 - len(tmp_ids))
            attn_mask[-1 - (MAX_LEN - 2 - len(tmp_ids)) : -1] = 0
        elif len(tmp_ids) > MAX_LEN - 2:
            last_available = MAX_LEN - 2 - (len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2)
            if t_start and t_end:
                if t_start >= last_available + 1 or t_end >= last_available:
                    t_start = 0
                    t_end = 1
            text = tokenizer.decode(tmp_ids[:last_available]) + '[SEP]' + x + '[CLS]' + y
            x_st = len(tmp_ids[:last_available]) + 2

        token_type_ids = torch.zeros(512)
        second_seq_len = len(tokenizer(x).input_ids) + len(tokenizer(y).input_ids) - 2
        token_type_ids[-second_seq_len:] = 1

        x_nd = x_st + len(tokenizer(x).input_ids) - 2
        y_st = x_nd + 1
        y_nd = y_st + len(tokenizer(y).input_ids) - 2

        ############# Assertion #############
        tmp_ids = tokenizer(text).input_ids
        if t_start != 0 and t_end != 1:
            try:
                if not tokenizer.decode(tmp_ids[t_start:t_end]).replace(' ', '') == tokenizer.decode(tokenizer(trigger).input_ids[1:-1]).replace(' ', '') and \
                        not tokenizer.decode(tokenizer(trigger).input_ids[1:-1]).replace(' ', '') in tokenizer.decode(tmp_ids[t_start:t_end]).replace(' ', ''):
                    raise NameError('Error from TRIGGER!')
            except:
                ipdb.set_trace()
        try:
            assert tokenizer.decode(tmp_ids[x_st:x_nd]).replace(' ', '') == tokenizer.decode(tokenizer(x).input_ids[1:-1]).replace(' ', '')
        except:
            ipdb.set_trace()
        try:
            assert tokenizer.decode(tmp_ids[y_st:y_nd]).replace(' ', '') == tokenizer.decode(tokenizer(y).input_ids[1:-1]).replace(' ', '')
        except:
            ipdb.set_trace()

        try:
            assert len(tmp_ids) == MAX_LEN
        except:
            ipdb.set_trace()
        #####################################
        t_len = t_end - t_start if len(trigger) > 0 else 0
        examples.append({
            'has_trigger': 1 if len(trigger) > 0 else 0,
            'trigger_len': t_len,
            'd': d,
            'x': x,
            'y': y,
            'text': text,
            'input_ids': tmp_ids,
            'attention_mask': attn_mask,
            'token_type_ids': token_type_ids,
            'label': int(label)-1,
            't_start': t_start,
            't_end': t_end,
            'trigger': trigger,
            'x_st': x_st,
            'x_nd': x_nd,
            'y_st': y_st,
            'y_nd': y_nd
        })
    with open(PATH[:-4] + '.pkl', 'wb') as fp:
        pickle.dump(examples, fp)
    print('cached!')

if __name__ == '__main__':
    preprocess('data/dev.txt', 'dev')
    preprocess('data/train.txt', 'train')
    preprocess('data/test.txt', 'test')
