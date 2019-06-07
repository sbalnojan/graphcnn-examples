# Easy AI

Easy AI is a set of packages and utilities to make it easy to use some
"standard" machine learning models already out there.

The infrastructure code is currently aimed at AWS as cloud provider.

Some examples included:

- code to setup a suitable EC2 instance for
  [PGANs](https://github.com/tkarras/progressive_growing_of_gans)
  with custom images.

## Overview of code base

```
.
├── README.md
├── data
│   └── cora
├── example_even_simpler_0.py   # Example on 3x3 matrix
├── example_simple_1.py         # Example on CORA data set
├── keras-deep-graph-learning   # Submodule used
│   ├── ...
└── requirements.txt            # Nec. dev requirements
```

## References

- https://github.com/tkipf/keras-gcn
- https://arxiv.org/abs/1609.02907
- http://tkipf.github.io/graph-convolutional-networks/
- https://github.com/vermaMachineLearning/keras-deep-graph-learning
