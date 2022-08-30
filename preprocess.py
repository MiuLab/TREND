from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import ipdb
import spacy
import ipdb
from contraction import contractions_dict
from config import args

import re
contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


MAX_LEN = 512
nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    text = nlp(text)
    return " ".join([(t.text) for t in text if t.pos_ not in ['PUNCT', 'SPACE'] or t.lemma_ == ':']).replace(" '", "'").replace(" n't", "n't")

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
    data = json.load(open(PATH, 'r'))
    examples = []
    duplicate = 0
    map = {}
    map1 = {'under500':0, 'over500': 0}
    trigger_location = {500: 0, 1000: 0, 1500: 0}
    for i in tqdm(range(len(data))):
        text_a = " ".join(data[i][0])
        text_a = expand_contractions(text_a)

        for j in range(len(data[i][1])):
            text_b = data[i][1][j]['x']
            text_b = expand_contractions(text_b)
            text_c = data[i][1][j]['y']
            text_c = expand_contractions(text_c)
            map[len(data[i][1][j]['rid'])] = \
                map.get(len(data[i][1][j]['rid']), 0) + 1
            for l in range(len(data[i][1][j]['rid'])):
                if l>0: duplicate += 1
                if mode == 'train' and l > 0:
                    break
                label = data[i][1][j]['rid'][l] - 1
                d, x, y = rename(text_a.lower(), text_b.lower(), text_c.lower())
                t_start = 0
                t_end = 1
                trigger = data[i][1][j]['t'][l].lower()
                trigger = expand_contractions(trigger)
                if len(trigger) > 0:
                    t_start = d.find(trigger)
                    t_start = len(tokenizer(d[:t_start]).input_ids) - 1
                    trigger_ids = tokenizer(trigger).input_ids[1:-1]
                    d_ids = tokenizer(d).input_ids
                    for k in range(len(d_ids)):
                        if d_ids[k : k + len(trigger_ids)] == trigger_ids:
                            t_start = k
                            break
                    try:
                        t_end = t_start + len(tokenizer(trigger).input_ids) - 2
                    except:
                        ipdb.set_trace()

                text = d + '[SEP]' + x + '[CLS]' + y
                attn_mask = torch.ones(512)

                tmp_ids = tokenizer(text).input_ids[1:-1]

                if len(tmp_ids) <= MAX_LEN:
                    map1['under500'] += 1
                else:
                    map1['over500'] += 1
                    if t_end <= 500:
                        trigger_location[500] += 1
                    elif t_end <= 1000:
                        trigger_location[1000] += 1
                    else:
                        trigger_location[1500] += 1

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
                distributions = [-100] * 15
                if len(trigger) > 0:
                    distributions[:t_end-t_start] = tmp_ids[t_start:t_end]

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
                    'label': label,
                    't_start': t_start,
                    't_end': t_end,
                    'trigger': trigger,
                    'x_st': x_st,
                    'x_nd': x_nd,
                    'y_st': y_st,
                    'y_nd': y_nd,
                    'distributions': distributions
                })
    with open(PATH[:-5] + '.pkl', 'wb') as fp:
        pickle.dump(examples, fp)
    print('cached!')
    print(f"duplicate: {duplicate}")
    print(f"stats: {map}")
    print(map1)
    print(f"trigger_location: {trigger_location}")

if __name__ == '__main__':
    preprocess('data/dev.json', 'dev')
    preprocess('data/train.json', 'train')
    preprocess('data/test.json', 'test')
