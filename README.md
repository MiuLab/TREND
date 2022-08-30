# TREND: Trigger-Enhanced Relation Extraction Network for Dialogues
- Paper Link: https://arxiv.org/pdf/2108.13811.pdf
<img width="1384" alt="截圖 2022-08-31 上午12 35 40" src="https://user-images.githubusercontent.com/101118634/187562077-b16e1b5e-a6e8-43f3-90e5-865ec9a33e68.png">

## Data
### data_dre/
- Train, Val, Test sets of DialogRE
### data/
- Train, Val, Test sets of DDRel

## Preprocessing
- Use preprocess*.py to get the corresponding preprocessed pickle files.

## Training
`python main.py --test_path [data, data_dre]/dev.pkl`

## Testing
`python test.py --test_path [data, data_dre]/test.pkl --load_path PATH/TO/MODEL`

## Reference
Please cite the following paper:
```
@inproceedings{lin2022trend,
  title={TREND: Trigger-Enhanced Relation Extraction Network for Dialogues},
  author={Lin, Po-Wei and Su, Shang-Yu and Chen, Yun-Nung},
  booktitle={Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  year={2022}
}
```
