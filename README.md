# Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics

<img src="https://github.com/bayesgroup/bayesgroup.github.io/blob/master/tqc/assets/tqc/main_exps_pytorch.svg">

This repository implements continuous reinforcement learning method TQC, described in paper ["Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"](https://arxiv.org/abs/2005.04269) on PyTorch.
Official implementation on Tensorflow can be found [here](https://github.com/bayesgroup/tqc). Source code is based on [TD3](https://github.com/sfujim/TD3), and we thank the authors for their efforts.

## Requirements

### Install MuJoCo

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 1.50 from the MuJoCo website. We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150`). Gym and MuJoCo 2.0 have integration bug, where Gym doesn't process contanct forces correctly for environments Humanoid and Ant.
Please use MuJoCo 1.5.

2. Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt:

### Create anaconda environment

Create anaconda environment from provided environment.yaml file:

```
cd ${SOURCE_PATH}
conda env create -f environment.yml
conda activate tqc
```

It essentially consists of ```gym==0.12.5, mujoco-py==1.50.1.68, pytorch==1.3.0, torchvision==0.2.1```.

Environment contains ```mujoco-py``` library which [may require](https://github.com/openai/mujoco-py) to install additional libraries depending on OS.

## Usage
Experiments on single environments can be run by calling:

```
python main.py --env Walker2d --top_quantiles_to_drop_per_net 2
```

Hyper-parameters can be modified with different arguments to main.py.

Number of atoms to remove for each environment:

| Environment        | top_quantiles_to_drop_per_net  |
| ------------- |:-------------:|
| Hopper           | 5 |
| HalfCheetah      | 0 |
| Walker2d         | 2 |
| Ant              | 2 |
| Humanoid         | 2 |

