# Brainseg

Code based on an implementation of TransUnet & UniMatch of brain organ segmentation。

## Preparation

### Installation

```bash
cd Brainseg
pip install -r requirements.txt
python setup.py install
```
### Dataset:
Trainging data is t1t2 weight data: train_t1t2.h5

### File:

```
├── [Brainseg]
    └──  train_t1t2.h5
```

## Start:
```
sbacth python3gpu.job train_match.py
```
