import numpy as np
import scipy.misc
import json
import os


# import the trainer class
from python.tensorflow_trainer import Trainer
from python.tensorflow_trainer_fusion import TrainerFusion

# import the models
import python.models.tensorflow_unet as model
import python.models.tensorflow_residual_fusion as model_fusion

# load the configuration file and define the variables
print("Loading configuration file")
import argparse
parser = argparse.ArgumentParser(description='Semantic3D')
parser.add_argument('--config', type=str, default="config.json", metavar='N',
help='config file')
args = parser.parse_args()
json_data=open(args.config).read()
config = json.loads(json_data)
imsize = config["imsize"]
directory = config["train_results_root_dir"]
dir_images = os.path.join(directory,config["images_dir"])
saver_directory_rgb = config["saver_directory_rgb"]
saver_directory_composite = config["saver_directory_composite"]
saver_directory_fusion = config["saver_directory_fusion"]
vgg_weight_init = config["vgg_weight_init"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
epoch_nbr = config["epoch_nbr"]
train_rgb = config["train_rgb"]
train_composite = config["train_composite"]
train_fusion = config["train_fusion"]
label_nbr = config["label_nbr"]
input_ch = config["input_ch"]

if train_rgb:
    # train the rgb model
    print("Training RGB")
    trainer = Trainer(model.model)
    trainer.train(
        imsize = imsize,
        batch_size=batch_size,
        input_ch = input_ch,
        epoch_nbr = epoch_nbr,
        net_weights_init = vgg_weight_init,
        dir_images = dir_images,
        saver_directory = saver_directory_rgb,
        images_root = "rgb",
        label_nbr = label_nbr,
        learning_rate = learning_rate,
        variable_scope="rgb")

if train_composite:
    # train the composite model
    print("Training composite")
    trainer = Trainer(model.model)
    trainer.train(
        imsize = imsize,
        batch_size=batch_size,
        input_ch = input_ch,
        epoch_nbr = epoch_nbr,
        net_weights_init = vgg_weight_init,
        dir_images = dir_images,
        saver_directory = saver_directory_composite,
        images_root = "composite",
        label_nbr = label_nbr,
        learning_rate = learning_rate,
        variable_scope="composite")

if train_fusion:
    # train the fusion model
    print("Training fusion")
    trainer = TrainerFusion(model.model, model.model, model_fusion.model)
    trainer.train(
            imsize = imsize,
            batch_size=batch_size,
            input_ch = input_ch,
            epoch_nbr = epoch_nbr,
            net_weights_init = None,
            dir_images = dir_images,
            saver_directory1= saver_directory_rgb,
            saver_directory2= saver_directory_composite,
            saver_directory = saver_directory_fusion,
            images_root1 = "rgb",
            images_root2 = "composite",
            label_nbr = label_nbr,
            learning_rate = learning_rate,
            variable_scope1="rgb",
            variable_scope2="composite",
            variable_scope_fusion="fusion")
