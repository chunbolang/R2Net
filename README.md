# Global Rectification and Decoupled Registration for Few-Shot Segmentation in Remote Sensing Imagery

This repository contains the source code for our paper "*Global Rectification and Decoupled Registration for Few-Shot Segmentation in Remote Sensing Imagery*" by Chunbo Lang, Gong Cheng, Binfei Tu, and Junwei Han.

> **Abstract:** *Few-shot segmentation (FSS), which aims to determine specific objects in the query image given only a handful of densely labeled samples, has received extensive academic attention in recent years. However, most existing FSS methods are designed for natural images, and few works have been done to investigate more realistic and challenging applications, *e.g.*, remote sensing image understanding. In such a setup, the complex nature of the raw images would undoubtedly further increase the difficulty of the segmentation task. To couple with potential inference failures, we propose a novel and powerful remote sensing FSS framework with global Rectification and decoupled Registration, termed R<sup>2</sup>Net. Specifically, a series of dynamically updated global prototypes are utilized to provide auxiliary non-target segmentation cues and to prevent inaccurate prototype activation resulting from the variability between query-support image pairs. The foreground and background information flows are then decoupled for more targeted and tailored object localization, avoiding unnecessary confusion from information redundancy. Furthermore, we impose additional constraints to promote the interclass separability and intraclass compactness. Extensive experiments on the standard benchmark iSAID-5<sup>i</sup> demonstrate the superiority of the proposed R<sup>2</sup>Net over state-of-the-art FSS models.*

## 🌳 Code Structure

```
├─R2Net
|   ├─test.py
|   ├─test.sh
|   ├─train.py
|   ├─train.sh
|   ├─train_base.py
|   ├─train_base.sh
|   ├─util
|   ├─model
|   |   ├─workdir
|   |   ├─util
|   |   ├─few_seg
|   |   |    └R2Net.py
|   |   ├─backbone
|   ├─lists
|   ├─initmodel
|   |     ├─PSPNet
|   ├─exp
|   ├─dataset
|   ├─config
├─data
|  ├─iSAID
|  |   ├─train.txt
|  |   ├─val.txt
|  |   ├─img_dir
|  |   ├─ann_dir
```

## 📝 Data Preparation

- Create a folder `data` at the same level as this repo in the root directory.

  ```
  cd ..
  mkdir data
  ```
  
- Download the iSAID dataset from our [[OneDrive]](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EbOLExyJqaFLquSK2F2oNGMBA3_c7qlttFm_tROxnsR9Cg?e=wqteBv) and put it in the `data` directory.

## ▶️ Getting Started

### Training base-learners (two options)

- Option 1: training from scratch

  Download the pre-trained backbones from [here](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EflpnBbWaftEum485cNq8v8BdSHiKvXLaX-dBBsbtdnCjg?e=FAxL2e) and put them into the `R2Net/initmodel` directory.
  ```
  train_base.sh
  ```
- Option 2: loading the trained models
  
  ```
  mkdir initmodel
  cd initmodel
  ```
  
  Put the provided [models](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/langchunbo_mail_nwpu_edu_cn/EXgoKTugPoJFhrJVd2gK8iUBaPXlYePQtOt0xpWS60qoLw?e=EJ9tEW) in the newly created folder `initmodel` and rename the downloaded file to `PSPNet`, *i.e.*, `R2Net/initmodel/PSPNet`.

### Training few-shot models

To train a model, run

```
train.sh
```

### Testing few-shot models

To evaluate the trained models, run

```
test.sh
```

## 🎉 Features

- [x] Distributed training (Multi-GPU)
- [x] Different dataset divisions
- [x] Multiple runs

## 📖 BibTex
If you find this repository useful for your publications, please consider citing our paper.

Our paper is under review...

## 👏 Acknowledgements
The project is based on [PFENet](https://github.com/dvlab-research/PFENet) and [mmseg](https://github.com/open-mmlab/mmsegmentation). Thanks for the authors for their efforts.

[BinfeiTu](https://github.com/Binfeitu) is the main contributor to this repository.
