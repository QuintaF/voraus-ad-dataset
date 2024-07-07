# Anomaly Detection with MVT-Flow

**IMPORTANT**: This readme is only helpful to the training on the new datasets(Pepper and SWaT). Before further reading be sure to check the [original README](originalREADME.md). Since none of the original files has been modified, instructions can be followed to the letter even after installing this repository. Check correctness for the path pointing to voraus dataset files in [train.py](train.py) at line 21 (before proceeding make sure that tests end successfully).

This project aims to tests the performance of the normalizing flow method MVT-Flow on datasets different from the one it has been designed on. More informations in the official paper ["The voraus-AD Dataset for Anomaly Detection
in Robot Applications"](https://arxiv.org/pdf/2311.04765.pdf).\
For a step-by-step configuration guide follow the [original README](originalREADME.md) guide.

## Usage: Train

Dataset files are searched starting from the *CWD*, which is expected to end as *".../voraus-ad-dataset/"*

### Voraus Training
```
usage: train.py
```
More info on the original README

### SWaT & Pepper Training

```
usage: train_custom.py [-h] [--seed SEED] [--dataset {pepper,swat,voraus}] [--ratio RATIO] [--normalize] [--downsample] [--plot]
```

Usage options:

```
optional arguments:
  -h, --help            show this help message and exit
  --seed SEED, -sd SEED
                        choose the seed for rng functions in the code
  --dataset {pepper,swat,voraus}, -dt {pepper,swat,voraus}
                        choose which dataset to load
  --ratio RATIO, -rt RATIO
                        a ratio between 0.4 and 0.8 for deciding how large the training set will be
  --normalize, -n       if set, data is normalized
  --downsample, -ds     if set, data is downsampled(only for SWaT dataset)
  --plot, -plt          if set, metric plots are shown(only for model evaluation)
```

### Default Execution
The default execution trains on pepper datasets with no normalization nor downsampling, the default seed is 177 and the normal data used for training constitutes 80% of the dataset(ratio=.8).

N.B. using voraus as dataset option for custom training does not work


## Usage: Evaluation
Dataset files are searched starting from the *CWD*, which is expected to end as *".../voraus-ad-dataset/"*

```
usage: train_custom.py [-h] [--seed SEED] [--dataset {pepper,swat,voraus}] [--ratio RATIO] [--normalize] [--downsample] [--plot]
```

Usage options:

```
optional arguments:
  -h, --help            show this help message and exit
  --seed SEED, -sd SEED
                        choose the seed for rng functions in the code
  --dataset {pepper,swat,voraus}, -dt {pepper,swat,voraus}
                        choose which dataset to load
  --ratio RATIO, -rt RATIO
                        a ratio between 0.4 and 0.8 for deciding how large the training set will be
  --normalize, -n       if set, data is normalized
  --downsample, -ds     if set, data is downsampled(only for SWaT dataset)
  --plot, -plt          if set, metric plots are shown(only for model evaluation)
```

### Default Execution
The default execution evaluates pepper dataset with no normalization nor downsampling, the default seed is 177 and the normal data used for evaluation constitutes 20% of the dataset(ratio=.8).

## Repository Structure
```
voraus-ad-dataset/ 
├── Dataset/
│   └── Voraus dataset files
│
├── Pepper & SWAT/
│   ├── Pepper normal and attack csv files
│   └── SWaT normal and attack files, both original xlsx and modified csv
│
├── tests/
│    └── read the original README
│
└── python files
```
----

## Hyperparameters


The parameters to configure the environment and those to modify the execution of the algorithm (located at the top of the training file after imports).

```python
# Define the model training configuration and hyperparameters.
configuration = Configuration(
    columns="machine",  #only for voraus dataset
    epochs=70,
    frequencyDivider=1, #only for voraus dataset
    trainGain=1.0,      #only for voraus dataset
    seed=177,
    batchsize=32,
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=8e-4,
    normalize=True,    
    pad=True,           #only for voraus dataset
    nHiddenLayers=0,
    scale=2,
    kernelSize1=13,
    dilation1=2,
    kernelSize2=1,
    dilation2=1,
    kernelSize3=1,
    dilation3=1,
    milestones=[11, 61],
    gamma=0.1,
)
```
    
