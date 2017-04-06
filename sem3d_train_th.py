import argparse
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
from random import  shuffle
from tqdm import *
import scipy.misc
from collections import OrderedDict

import python.models.pytorch_segnet as network

# model from
#https://github.com/jcjohnson/pytorch-vgg

label_nbr = 11
model_path = "/data/data/caffemodels/vgg16-00b39a1b.pth"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Semantic3D')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-dir', type=str, default=".", metavar='N',
                    help='save model directory')
parset.add_argument('--image-type', type=str, default="rgb", metavar='N',
                    help='image type to train on (default rgb)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


images_root = args.image_type
dir_images = "/data/semantic_3d/test_train_2/images_test_class"


# set the seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


##########
# Create tuncated VGG model
print("Creating model...")
model = network.SegNet(3, label_nbr)
if args.cuda:
    model.cuda()
else:
    model.float()
model.eval()
print("---> done")


# define the optimizer
print("Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print("---> done")


# create the list of images in the folder
print("Generating file...")
directory = os.path.join(dir_images, images_root)
directory_labels = os.path.join(dir_images, "labels/")
files = []
for file in os.listdir(directory_labels):
    if file.endswith(".npz"):
        file = file.split(".")[:-1]
        file = ".".join(file)
        files.append(file)
print("---> done")

def train(epoch):
    model.train()
    shuffle(files)

    # create the batches
    batches = [files[i:i + args.batch_size] for i in range(0, len(files), args.batch_size)]
    batches = batches[:-1] # remove last batch (potentially not the same size)

    # containers
    batch_np= np.zeros((args.batch_size,3, 224, 224), dtype=float)
    labels_np = np.zeros((args.batch_size,224, 224), dtype=int)

    # define loss
    loss = nn.NLLLoss2d().cuda()
    softmax = nn.Softmax2d()

    total_loss = 0

    # iteration over the batches
    for batch_idx,batch_files in enumerate(tqdm(batches)):
        for im_id in range(len(batch_files)):
            batch_np[im_id] = scipy.misc.imread(os.path.join(directory, batch_files[im_id]+".png")).transpose()
            labels_np[im_id] = np.load(os.path.join(directory_labels, batch_files[im_id]+".npz"))["arr_0"].transpose()
        batch_np /= 255 #normalize the input

        # variables
        batch_th = Variable(torch.Tensor(batch_np))
        target_th = Variable(torch.LongTensor(labels_np))
        if args.cuda:
            batch_th = batch_th.cuda()
            target_th = target_th.cuda()

        optimizer.zero_grad()
        output, temp = model(batch_th)

        # compute loss
        output = softmax(output)
        l_ = loss(output, target_th)
        l_.cuda()
        l_.backward()

        # update weights
        optimizer.step()

        total_loss += l_.cpu().data.numpy()[0]

    return total_loss/len(batches)

# create corresponding directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

f = open(os.path.join(args.save_dir, "logs.txt"), "w")
for epoch in range(1, args.epochs + 1):
    print("Epoch "+str(epoch))
    train_loss = train(epoch)
    print("Train loss: "+str(train_loss))

    f.write(str(epoch)+" ")
    f.write(str(train_loss)+" ")
    f.write("\n")

    # save the model state dict
    if(epoch % 10 == 0):
        with open(os.path.join(args.save_dir, "model_state.pth"), 'wb') as f:
            torch.save(model.state_dict(), f)
f.close()
