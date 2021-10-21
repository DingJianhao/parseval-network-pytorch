import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def orthogonal_retraction(module, beta=0.002):
    with torch.no_grad():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if isinstance(module, nn.Conv2d):
                weight_ = module.weight.data
                sz = weight_.shape
                weight_ = weight_.reshape(sz[0],-1)
                rows = list(range(module.weight.data.shape[0]))
            elif isinstance(module, nn.Linear):
                if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0], -1)
                    rows = list(range(module.weight.data.shape[0]))
                else:
                    rand_rows = np.random.permutation(module.weight.data.shape[0])
                    rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                    weight_ = module.weight.data[rows,:]
                    sz = weight_.shape
            module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)


class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out


def convex_constraint(module):
    with torch.no_grad():
        if isinstance(module, ConvexCombination):
            comb = module.comb.data
            alpha = torch.sort(comb, descending=True)[0]
            k = 1
            for j in range(1,module.n+1):
                if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
                    k = j
                else:
                    break
            gamma = (torch.sum(alpha[:k]) - 1)/k
            module.comb.data -= gamma
            torch.relu_(module.comb.data)


if __name__ == "__main__":
    pass

    # import torch.optim as optim
    # net = ConvexCombination(10)
    # opt = optim.SGD(net.parameters(), lr=0.0001)
    # xs = [ torch.rand(2,2) for _ in range(10)]
    # for i in range(10):
    #     xs[i].requires_grad = True
    # y = net(*xs)
    # loss = y.sum()
    # loss.backward()
    # opt.step()
    # # print(net.comb.grad)
    # convex_constraint(net)


    # import torch.optim as optim
    #
    # x = torch.rand(4,3,10,10)
    # # net = nn.Conv2d(3,64,5)
    # net = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(300, 20)
    # )
    #
    # opt = optim.SGD(net.parameters(), lr=0.0001)
    # y = net(x)
    # opt.zero_grad()
    # loss = y.sum()
    # loss.backward()
    # opt.step()
    # orthogonal_retraction(net[1])
