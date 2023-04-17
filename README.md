# Medical Symptom Detection in Intelligent Pre-Consultation using Bi-directional Hard-Negative Noise Contrastive Estimation 

This repository holds the source code for our KDD 2022 paper titled: [Paper](https://dl.acm.org/doi/pdf/10.1145/3534678.3539124)

<!-- Thanks for your interest in our repo! -->

## Overview

Our approach to symptom detection in medical queries within intelligent pre-consultation (IPC) takes into consideration the colloquial language typically used. Rather than treating the task as a medical concept normalization problem, we approach it as a retrieval problem. To accomplish this, we introduce a bi-directional hard-negative enforced noise contrastive estimation model (Bi-hardNCE) for symptom detection. We demonstrate that our proposed model surpasses popular retrieval models as well as the current state-of-the-art NCE model which uses in-bath negatives.

![](img/sym.png)

## Getting Started

### Requirements
- python3.6
- pip install -r requirements.txt

### Downloading Chinese Bert
python downlond_chinese_bert.py 

### Training
sh bihdnce.sh
