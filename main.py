import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable
import parseval as pars

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--net_type', default='wide-resnet', type=str)
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--widen_factor', default=10, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--parseval', '-p', action='store_true')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    trainset = torchvision.datasets.CIFAR10(root=cf.dataset_dir[args.dataset], train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=cf.dataset_dir[args.dataset], train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    trainset = torchvision.datasets.CIFAR100(root=cf.dataset_dir[args.dataset], train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=cf.dataset_dir[args.dataset], train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Return network & file name
def getNetwork(args):
    net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, convex_combination=args.parseval)
    file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    return net, file_name

# Test only option
# if (args.testOnly):
#     print('\n[Test Phase] : Model setup')
#     assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
#     _, file_name = getNetwork(args)
#     checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
#     net = checkpoint['net']
#
#     if use_cuda:
#         net.cuda()
#         net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#         cudnn.benchmark = True
#
#     net.eval()
#     net.training = False
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             if use_cuda:
#                 inputs, targets = inputs.cuda(), targets.cuda()
#             inputs, targets = Variable(inputs), Variable(targets)
#             outputs = net(inputs)
#
#             _, predicted = torch.max(outputs.data, 1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.data).cpu().sum()
#
#         acc = 100.*correct/total
#         print("| Test Result\tAcc@1: %.2f%%" %(acc))
#
#     sys.exit(0)


# Model
# if args.resume:
#     # Load checkpoint
#     print('| Resuming from checkpoint...')
#     assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
#     _, file_name = getNetwork(args)
#     checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
#     net = checkpoint['net']
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
# else:
print('| Building net type [' + args.net_type + ']...')
net, file_name = getNetwork(args)
net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    global args
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    if args.parseval:
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=0)
    else:
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        if args.parseval:
            for m in net.modules():
                pars.orthogonal_retraction(m)
                pars.convex_constraint(m)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test(epoch):
    global best_acc, writer
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        writer.add_scalar('test_acc', acc.item(), epoch)

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # save_point = './checkpoint/'+args.dataset+os.sep
            # if not os.path.isdir(save_point):
            #     os.mkdir(save_point)
            # torch.save(state, save_point+file_name+'.t7')
            best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

writer = SummaryWriter()

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
