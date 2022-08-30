# TREND: Trigger-Enhanced Relation Extraction Network for Dialogues
- Paper Link: https://arxiv.org/pdf/2108.13811.pdf
[model.pdf](https://github.com/MiuLab/TREND/files/9457158/model.pdf)

## Data
### data_dre/
- DialogRE
### data/
- DDRel

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
