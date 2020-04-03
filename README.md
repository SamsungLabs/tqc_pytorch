# Controlling Overestimation Bias with Truncated Mixture of ContinuousDistributional Quantile Critics

PyTorch implementation of Truncated Quantile Critics (TQC).

### Requirements

Create anaconda environment from provided environment.yaml file:

```
conda env create -f environment.yml 
```

It essentially consists of ```gym==0.12.5, mujoco-py==1.50.1.68, pytorch=1.3.0, torchvision=0.2.1```.

Environment contains ```mujoco-py``` library which [may require](https://github.com/openai/mujoco-py) to install additional libraries depending on OS.

### Usage
Experiments on single environments can be run by calling from created environment:

```
conda activate tqc
python main.py --env HalfCheetah-v2
```

Hyper-parameters can be modified with different arguments to main.py.
