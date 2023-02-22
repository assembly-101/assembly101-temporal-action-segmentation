# assembly101-temporal-action-segmentation
This repository contains code and model for the Temporal Action Segmentation benchmark of Assembly101.

If you use our dataset and model, kindly cite:
```
@article{sener2022assembly101,
    title = {Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities},
    author = {F. Sener and D. Chatterjee and D. Shelepov and K. He and D. Singhania and R. Wang and A. Yao},
    journal = {CVPR 2022},
}

@article{singhania2021coarse,
  title={Coarse to fine multi-resolution temporal convolutional network},
  author={Singhania, Dipika and Rahaman, Rahul and Yao, Angela},
  journal={arXiv preprint arXiv:2105.10859},
  year={2021}
}
```

## Contents
* [Overview](#overview)
* [Data](#data)
* [Training](#training)
* [Evaluate](#evaluate)

## Overview

This repository provides codes to train and validate for the task of coarse temporal action segmentation of the Assembly101 dataset. [C2F-TCN](https://github.com/dipika-singhania) is used here. 

## Data

Per-frame features are required as input. TSM (8-frame input) has been used for extracting 2048-D per-frame features which can be downloaded from our [Gdrive](https://drive.google.com/drive/folders/1nh8PHwEw04zxkkkKlfm4fsR3IPEDvLKj). Please follow [this](https://github.com/assembly-101/assembly101-download-scripts) for requesting drive access to download the `.lmdb` TSM features.

The action segmentation annotations and can be found [here](https://github.com/assembly-101/assembly101-annotations). Only the coarse-annotations are used.

Run [`data/data_stat.py`](data/data_stat.py) to generate data statistics for each video after downloading the `.lmdb` features.

```bash
python data/data_stat.py lmdb_path
```

## Training

To train our model, run

```bash
python main.py --action train --feature_path lmdb_path --split train
```

set `--split train_val` to use both train and val data for training.

## Evaluate

To evaluate our model, run  

```bash
python main.py --action predict --feature_path lmdb_path --test_aug 0
```

|  Split     | test_aug | F1@10 | F1@25 | F1@50 | Edit | MoF|
|:-------:|:--------:|:--------:|:---------:|:---------:|:--------:|:--------:|
| **Train** |    False   |   33.3   |   28.6    |    20.6   |   31.7   |    37.8   |

set `--test_aug 1` to use data augmentation for evaluation.

The pre-trained model can be found [here](https://drive.google.com/file/d/1Y_vibCBhq6w3hN8ufi7UAXmkxorxilV7/view?usp=sharing)


## License
Assembly101 is licensed by us under the Creative Commons Attribution-NonCommerial 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc/4.0/). The terms are :
- **Attribution** : You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** : You may not use the material for commercial purposes.