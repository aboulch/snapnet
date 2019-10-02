import numpy as np
import os
import scipy.misc
import pickle
from tqdm import *
import shutil


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

# from plyfile import PlyData, PlyElement
# import pointcloud_tools.lib.python.PcTools as PcTls

class SnapNetDataset(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, filenames,
                image_directory1,
                image_directory2,
                dir_views):
        """Init function."""

        self.filenames = filenames
        self.image_directory1 = image_directory1
        self.image_directory2 = image_directory2
        self.dir_views = dir_views

    def __len__(self):
        """Length."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Get item."""
        # expect a global variable called

        im1 = np.array(Image.open(os.path.join(self.image_directory1, self.filenames[index]+".png")))
        im2 = np.array(Image.open(os.path.join(self.image_directory2, self.filenames[index]+".png")))
        indices = np.load(os.path.join(self.dir_views, self.filenames[index]+".npz"))["arr_0"]

        im1 = im1.transpose(2,0,1).astype(np.float32) / 255 - 0.5
        im2 = im2.transpose(2,0,1).astype(np.float32) / 255 - 0.5
        indices = indices.astype(np.int64)
        
        im1 = torch.from_numpy(im1).float()
        im2 = torch.from_numpy(im2).float()
        indices = torch.from_numpy(indices).long()

        return im1, im2, indices


class BackProjeter:

    def __init__(self, model_function1, model_function2, model_fusion):
            self.model_function1 = model_function1
            self.model_function2 = model_function2
            self.model_function_fusion = model_fusion


    def backProj(self,
        filename,
        label_nbr,
        dir_mesh,
        dir_images,
        imsize,
        input_ch,
        batch_size,
        saver_directory1,
        saver_directory2,
        saver_directoryFusion,
        images_root1,
        images_root2,
        variable_scope1,
        variable_scope2,
        variable_scope_fusion):

        # load mesh
        faces = np.loadtxt(os.path.join(dir_mesh, filename+"_voxels_faces.txt")).astype(int)
        vertices = np.loadtxt(os.path.join(dir_mesh,filename+"_voxels.txt"))[:,:3]


        # dir views
        dir_views = os.path.join(dir_images, "views")

        # load model
        net1 = self.model_function1(input_ch, label_nbr)
        net1.load_state_dict(torch.load(os.path.join(saver_directory1, "state_dict.pth")))
        net2 = self.model_function2(input_ch, label_nbr)
        net2.load_state_dict(torch.load(os.path.join(saver_directory2, "state_dict.pth")))
        net_fusion = self.model_function_fusion(2*64, label_nbr)
        net_fusion.load_state_dict(torch.load(os.path.join(saver_directoryFusion, "state_dict.pth")))
        net1.cuda()
        net2.cuda()
        net_fusion.cuda()
        net1.eval()
        net2.eval()
        net_fusion.eval()

        # create the dataloader
        im_images1_dir = os.path.join(dir_images, images_root1)
        im_images2_dir = os.path.join(dir_images, images_root2)
        image_files = [os.path.splitext(f)[0] for f in os.listdir(im_images1_dir) if os.path.isfile(os.path.join(im_images1_dir, f)) and ".png" in f]

        ds = SnapNetDataset(image_files, im_images1_dir, im_images2_dir, dir_views)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2) 


        # create score matrix
        scores = np.zeros((vertices.shape[0],label_nbr))
        counts = np.zeros(vertices.shape[0])
        with torch.no_grad():

            t = tqdm(data_loader, ncols=100, desc=f"Test")
            for inputs1, inputs2, indices_th in t:

                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()

                outputs1, features1 = net1(inputs1, return_features=True)
                outputs2, features2 = net2(inputs2, return_features=True)
                outputs = net_fusion(outputs1, outputs2, features1, features2)

                outputs_np = outputs.cpu().detach().numpy().transpose(0,2,3,1)
                indices_np = indices_th.cpu().detach().numpy()

                for im_id in range(indices_np.shape[0]):

                    preds_ = outputs_np[im_id]
                    indices = indices_np[im_id]
                    indices = indices.reshape((-1))
                    indices[indices>faces.shape[0]] = 0
                    preds_ = preds_.reshape((-1, preds_.shape[2]))
                    scores[faces[indices-1][:,0]] += preds_
                    scores[faces[indices-1][:,1]] += preds_
                    scores[faces[indices-1][:,2]] += preds_

                    counts[faces[indices-1][:,0]] += 1
                    counts[faces[indices-1][:,1]] += 1
                    counts[faces[indices-1][:,2]] += 1

            counts[counts ==0] = 1
            scores /= counts[:,None]

            # force 0 label to unseen vertices
            scores[scores.sum(axis=1)==0][0] = 1

            scores = scores.argmax(axis=1)

            self.scores = scores

    def saveScores(self,filename):
        np.savetxt(filename, self.scores)

    def saveSemantizedCloud(self, filename,
        dir_mesh,
        save_dir):

        vertices = np.loadtxt(os.path.join(dir_mesh,filename+"_voxels.txt"))[:,:4]
        vertices[:,3] = self.scores
        np.savetxt(os.path.join(save_dir, filename+"_output.txt"), vertices, fmt=('%.3f', '%.3f', '%.3f', '%i'))
