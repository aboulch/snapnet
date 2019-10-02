import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from random import shuffle
import scipy.misc
import shutil

from tqdm import tqdm

import torch.utils.data as data
import random
import numbers

from PIL import Image

import python.pytorch.metrics as metrics
from sklearn.metrics import confusion_matrix

class SnapNetDataset(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, filenames,
                image_directory,
                training=False,
                label_directory=None):
        """Init function."""

        self.filenames = filenames
        self.image_directory = image_directory
        self.training = training
        self.label_directory = label_directory

    def __len__(self):
        """Length."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        im = np.array(Image.open(os.path.join(self.image_directory, self.filenames[index]+".png")))
        labels = None
        if self.training:
            labels = np.load(os.path.join(self.label_directory, self.filenames[index]+".npz"))["arr_0"]

        # data augmentation
        if self.training:
            if random.randint(0,1):
                im = im[::-1]
                labels = labels[::-1]
            if random.randint(0,1):
                im = im[:,::-1]
                labels = labels[:,::-1]

        im = im.transpose(2,0,1).astype(np.float32) / 255 - 0.5
        labels = labels.astype(np.int64)
        
        im = torch.from_numpy(im).float()

        if self.training:
            labels = torch.from_numpy(labels).long()
            return im, labels
        else:
            return im

class Trainer:

    def __init__(self, model_function):
            self.model_function = model_function

    def train(self,
        imsize,
        batch_size,
        input_ch,
        epoch_nbr,
        net_weights_init,
        dir_images,
        saver_directory,
        images_root,
        label_nbr,
        learning_rate,
        variable_scope="s"):


        # create model
        net = self.model_function(input_ch, label_nbr)
        net.cuda()

        # create the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        # create the dataloader
        im_labels_dir = os.path.join(dir_images, "labels")
        im_images_dir = os.path.join(dir_images, images_root)
        image_files = [os.path.splitext(f)[0] for f in os.listdir(im_labels_dir) if os.path.isfile(os.path.join(im_labels_dir, f)) and ".npz" in f]

        ds = SnapNetDataset(image_files, im_images_dir, True, im_labels_dir)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2) 

        os.makedirs(saver_directory, exist_ok=True)
        logs = open(os.path.join(saver_directory, "logs.txt"), "w")
        for epoch in range(epoch_nbr):
            
            cm = np.zeros((label_nbr, label_nbr))
            t = tqdm(data_loader, ncols=100, desc=f"Epoch {epoch}")

            for inputs, targets in t:

                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = net(inputs)
                optimizer.zero_grad()
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                outputs_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                targets_np = targets.cpu().numpy()

                cm += confusion_matrix(targets_np.ravel(), 
                                outputs_np.ravel(),
                                labels=list(range(label_nbr)))

                oa = metrics.stats_overall_accuracy(cm)
                avIoU = metrics.stats_iou_per_class(cm)[0]

                t.set_postfix(OA=f"{oa:.3f}", AvIOU=f"{avIoU:.3f}")

            os.makedirs(saver_directory, exist_ok=True)
            oa = metrics.stats_overall_accuracy(cm)
            avIoU = metrics.stats_iou_per_class(cm)[0]
            logs.write(f"{oa:.3f} {avIoU:.3f}\n")
            torch.save(net.state_dict(), os.path.join(saver_directory, "state_dict.pth"))
