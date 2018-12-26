
# Week 46: DropActivation: Implicit Parameter Reduction and Harmonic Regularization


[Original paper](https://arxiv.org/abs/1811.05850), Submitted on 14 Nov 2018

![intro](https://github.com/jguillaumin/one_week_one_paper/raw/master/week_46_drop_activation/plot_acc.png)

DropActivation is a new random activation function which reconciles Dropout and Batch Normalization (cf [Understanding the
disharmony between dropout and batch normalization by variance shift](https://arxiv.org/abs/1801.05134)). 

I implemented this new layer into Keras, and also [Randomized-ReLU]((https://arxiv.org/abs/1505.00853))


## Implemented features

**Note**: `ìmport keras` vs `import tensorflow.keras` 

I don't use `keras` python package, I use `tensorflow.keras` ! 
(I did not use Keras for 2 years ... it's so misleading to have Keras within TensorFlow now). 
So If you use raw python package `keras`, please be sure that the version of `keras` is the same as the version of Keras
in `tensorflow`. Today (Nov 2018), if you `pip install keras` it will be `keras.__version__==2.2.0` while the Keras branch in TensorFlow
is `tensorflow.keras.__version__==2.1.6` ...

If necessary, I will release a pure Keras version of the new layers.



**Implemented features:**

- `DropActivation` layer
- `RandomizedReLU` layer ([Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853))
- ResNet-56 on CIFAR-10 with Keras (TF backend), with MomentumSGD (0.9)
- data augmentation: random crop, horizontal flips and per sample standardization
- 3 notebooks (same code, just different networks) with seeded initialization (same initial random weights)


## Drop Activation: new activation layer which combines Dropout and ReLU (and it's compatible with BatchNorm!)

DropActivation combines ReLU and Dropout. 
In training mode, the activation function is random. Like relu, if the neural activation is positive, th identity function is used. 
If the input is negative, with a proba of p (p=0.95), the output is zero (like with ReLU). But with a proba 0.05,
the identity function is used.

So here, we switch randomly between ReLU(95%) and identity mapping(5%) function. 

At testing time, we use a LeakyReLU activation (deterministic) with a slope 1/(1-p) ! 


![formula](https://github.com/jguillaumin/one_week_one_paper/raw/master/week_46_drop_activation/formula.png)

## Comparision: ReLU, Drop Activation and Randomized ReLU

Here, a short comparision of ReLU, DropActivation and Randomized-ReLU

![table](https://github.com/jguillaumin/one_week_one_paper/raw/master/week_46_drop_activation/table.png)


## Results (from my code)

Model & training configuration:
- ResNet56 for CIFAR10 (no bottleneck block)
- L2 regularization only on kernels (conv and final dense layer) with weight `0.0002`
- optimization with SGD and Momentum (`0.9`)
- learning rate scheduling : 0.1, 0.01, 0.001 and 0.0001 (changes at epochs 91, 136 and 182)
- `batch_size=128` and `epochs=200`
- Data augmentation (only train, no test time augmentation): sample wise normalization (mean and std), 
    random crop (5 pixels), random horizontal flips (no vertical)
- training set vs validation : 80/20 % of initial training set (shuffle and stratified split)
- test set from CIFAR-10 as final test set !


Activation function | ACC-validation | ACC-test (generalization gap)
------------------- | ---------------- | ----------
Relu | 92.47 | 92.58 (**+ 0.11**)
Dropout with ReLU | 90.04 | 89.92 (- 0.12)
Randomized ReLU | 91.18 | 90.65 (- 0.53)
Drop Activation | **93.36** | **93.27** (- 0.09) 