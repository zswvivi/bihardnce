# Medical Symptom Detection in Intelligent Pre-Consultation using Bi-directional Hard-Negative Noise Contrastive Estimation 

This repository holds the source code for our KDD 2022 paper titled: [Paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539124)

<!-- Thanks for your interest in our repo! -->

## Overview

Our approach to symptom detection in medical queries within intelligent pre-consultation (IPC) takes into consideration the colloquial language typically used. Rather than treating the task as a medical concept normalization problem, we approach it as a retrieval problem. To accomplish this, we introduce a bi-directional hard-negative enforced noise contrastive estimation model (Bi-hardNCE) for symptom detection. We demonstrate that our proposed model surpasses popular retrieval models as well as the current state-of-the-art NCE model which uses in-bath negatives.

![](img/sym.png)


### Train Bi-hardNCE
In the following section, we describe how to train a Bi-hardNCE model by using our code.

### Requirements
- python3.6
- pip install -r requirements.txt

### Downloading Chinese Bert
```bash
python download_chinese_bert.py 
```
 
### Training

The training process for Bi-hardNCE differs slightly from traditional deep learning training. After each epoch, the model performs a prediction on the validation data to adjust the threshold for hard-negative mining. It then includes new hard negatives for each training instance.

```bash
sh bihdnce.sh
```

### Evaluation
```bash
python Evaluation.py
```

## Citation

Please cite our paper if you use Bi-hardNCE in your work:

```bibtex
@inproceedings{zhang2022medical,
  title={Medical Symptom Detection in Intelligent Pre-Consultation using Bi-directional Hard-Negative Noise Contrastive Estimation},
  author={Zhang, Shiwei and Sun, Jichao and Huang, Yu and Ding, Xueqi and Zheng, Yefeng},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4551--4559},
  year={2022}
}
```

