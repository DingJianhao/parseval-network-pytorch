# parseval-network-pytorch:
# Implementation of "Parseval networks: improving robustenss to adversarial examples"

This is an implementation of ["Parseval networks: improving robustenss to adversarial examples"](https://arxiv.org/abs/1704.08847).

## Requirements
- Install pytorch (mine is v1.7.1)

See [Pytorch.org](https://pytorch.org/) for more installation information.

## Run

To run training of wrn-28-10 with parseval net
```
python main.py --lr 0.1 --net-type wide_resnet --depth 28 --widen_factor 10 --dropout 0.3 --dataset cifar10 --parseval
```

If you want to run vanilla net work, please ignore --parseval

```
python main.py --lr 0.1 --net-type wide_resnet --depth 28 --widen_factor 10 --dropout 0.3 --dataset cifar10
```

## Details

The network used in the code is wideresnet-28-10. Model used here refers one network implementation from [this link](https://github.com/meliketoy/wide-resnet.pytorch).
If "--parseval", the training will actavate the use of orthogonal_retraction and convex_constraint in parseval_net, and the weight decay is set to zero (default 5e-4).

## Results

Test accuracy:

![Test_acc!](test_acc.svg")

The orange line is the result of the above first script, while the blue line is the result of the above first script.

