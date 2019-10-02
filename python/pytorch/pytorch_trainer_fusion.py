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
                image_directory1,
                image_directory2,
                training=False,
                label_directory=None):
        """Init function."""

        self.filenames = filenames
        self.image_directory1 = image_directory1
        self.image_directory2 = image_directory2
        self.training = training
        self.label_directory = label_directory

    def __len__(self):
        """Length."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        im1 = np.array(Image.open(os.path.join(self.image_directory1, self.filenames[index]+".png")))
        im2 = np.array(Image.open(os.path.join(self.image_directory2, self.filenames[index]+".png")))
        labels = None
        if self.training:
            labels = np.load(os.path.join(self.label_directory, self.filenames[index]+".npz"))["arr_0"]

        # data augmentation
        if self.training:
            if random.randint(0,1):
                im1 = im1[::-1]
                im2 = im2[::-1]
                labels = labels[::-1]
            if random.randint(0,1):
                im1 = im1[:,::-1]
                im2 = im2[:,::-1]
                labels = labels[:,::-1]

        im1 = im1.transpose(2,0,1).astype(np.float32) / 255 - 0.5
        im2 = im2.transpose(2,0,1).astype(np.float32) / 255 - 0.5
        labels = labels.astype(np.int64)
        
        im1 = torch.from_numpy(im1).float()
        im2 = torch.from_numpy(im2).float()

        if self.training:
            labels = torch.from_numpy(labels).long()
            return im1, im2, labels
        else:
            return im1, im2


class TrainerFusion:

    def __init__(self, model_function1, model_function2, model_function_fusion):
            self.model_function1 = model_function1
            self.model_function2 = model_function2
            self.model_function_fusion = model_function_fusion

    def train(self,
        imsize,
        batch_size,
        input_ch,
        epoch_nbr,
        net_weights_init,
        dir_images,
        saver_directory1,
        saver_directory2,
        saver_directory,
        images_root1,
        images_root2,
        label_nbr,
        learning_rate,
        variable_scope1,
        variable_scope2,
        variable_scope_fusion):


        # create models
        net1 = self.model_function1(input_ch, label_nbr)
        net1.load_state_dict(torch.load(os.path.join(saver_directory1, "state_dict.pth")))
        net2 = self.model_function2(input_ch, label_nbr)
        net2.load_state_dict(torch.load(os.path.join(saver_directory2, "state_dict.pth")))
        net_fusion = self.model_function_fusion(2*64, label_nbr)
        net1.cuda()
        net2.cuda()
        net_fusion.cuda()
        net1.eval()
        net2.eval()

        # create the optimizer
        optimizer = torch.optim.Adam(net_fusion.parameters(), lr=learning_rate)

        # create the dataloader
        im_labels_dir = os.path.join(dir_images, "labels")
        im_images1_dir = os.path.join(dir_images, images_root1)
        im_images2_dir = os.path.join(dir_images, images_root2)
        image_files = [os.path.splitext(f)[0] for f in os.listdir(im_labels_dir) if os.path.isfile(os.path.join(im_labels_dir, f)) and ".npz" in f]

        ds = SnapNetDataset(image_files, im_images1_dir, im_images2_dir, True, im_labels_dir)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2) 

        os.makedirs(saver_directory, exist_ok=True)
        logs = open(os.path.join(saver_directory, "logs.txt"), "w")
        for epoch in range(epoch_nbr):
            
            cm = np.zeros((label_nbr, label_nbr))
            t = tqdm(data_loader, ncols=100, desc=f"Epoch {epoch}")

            for inputs1, inputs2, targets in t:

                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()
                targets = targets.cuda()

                with torch.no_grad():
                    outputs1, features1 = net1(inputs1, return_features=True)
                    outputs2, features2 = net2(inputs2, return_features=True)


                outputs = net_fusion(outputs1, outputs2, features1, features2)
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
            torch.save(net_fusion.state_dict(), os.path.join(saver_directory, "state_dict.pth"))
