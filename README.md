# A recurrent state-space framework for modeling neural dynamics during decision-making

This package implements a wide class of decision-making models of neural activity in a unifying modeling framework. It includes the drift-diffusion model, one- and multi-dimensional accumulation to boundary models, the ramping model, and hybrid models with both ramping and discrete jumping components. The models are implemented by constraining the parameters of recurrent switching state-space models. The code extends the state-space modeling package [SSM](https://github.com/slinderman/ssm). 

Details of the modeling methods are in this paper:

Zoltowski, David M., Jonathan W. Pillow, and Scott W. Linderman. "[Unifying and generalizing models of neural dynamics during decision-making](https://arxiv.org/abs/2001.04571)." arXiv preprint arXiv:2001.04571 (2020).

# Installation

```
git clone https://github.com/davidzoltowski/ssmdm.git
cd ssmdm
pip install -e .
```
