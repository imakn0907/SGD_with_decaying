'''MNIST for the train accuracy of stop condition '''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='CIFAR-10 for the test accuracy of stop condition')
parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
parser.add_argument('--batchsize', default=256, type=int, help='training batch size')
parser.add_argument('--optimizer',default="sgd", type=str, help='[momentum,sgd,adam,rmsgrad,adamw]')
parser.add_argument('--use_wandb', default=False, type=str, help='Set to True if using wandb.')

#Comment out the following code when you don't use decaying 1~3.
#parser.add_argument('--decaynumber', default=1, type=int, help='decaysing learning rate number')

args = parser.parse_args()

#Comment out the following code when you don't use decaying 4.
#parser.add_argument('--stepsize', default=int(60000 * 20/args.batchsize), type=int, help='step size')
#args = parser.parse_args()


if args.use_wandb:
    wandb_project_name = "×××××"
    wandb_exp_name = f"×××××"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "×××××")
    wandb.init(settings=wandb.Settings(start_method='fork'))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = ResNet18_for_m()
#net = ResNet34_for_m()
#net = ResNet50_for_m()
#net = WideResNet28_10_for_m()

net = net.to(device)
if device == 'cuda:0':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#definition of decaying 1~3
def func(steps):
  b = steps + 1
  b = b ** (0.25 * args.decaynumber)
  return 1/b


criterion = nn.CrossEntropyLoss()
if args.optimizer == "momentum":
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)                 #Momentum
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))                           #Adam
elif args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)                                   #SGD
elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9)                                  #RMSProp      
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999))                          #AdamW

#Designation of decreasing step size.
#Comment out the following decaying number's code that you don't use.
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func) #decaying1~3
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=0.50)     #decaying4

# Training
steps = 0

def train(epoch):
    net.train()
    break_train_acc=90.0
    train_loss = 0
    correct = 0
    total = 0
    global steps
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        train_acc=correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #Update decaying step size.
        #Comment out the following code when using constant stepsize.
        '''last_lr = scheduler.get_last_lr()[0]
        if args.use_wandb:
            wandb.log({'last_lr': last_lr})
        scheduler.step()'''

        if train_acc > break_train_acc:
            print(train_acc,steps)

            if args.use_wandb:
                wandb.log({'loss':train_loss/(batch_idx+1),
                           'train_acc':train_acc,
                           'steps':steps})
            
            sys.exit()
        else:
            steps+=1
            if args.use_wandb:
                wandb.log({'loss':train_loss/(batch_idx+1),
                           'train_acc':train_acc,
                           'steps':steps})

    
def test(epoch):
    global best_acc
    net.eval()
    #Designation of the value of stop condition.
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    test_acc = 100.*correct/total
    if args.use_wandb:
        wandb.log({'test_acc': test_acc})

for epoch in range(start_epoch, start_epoch+200):
    print('\nEpoch: %d' % epoch)
    train(epoch)
    test(epoch)