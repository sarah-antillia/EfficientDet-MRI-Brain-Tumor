# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# split_train.py
# 2023/05/18 to-arai antillia.com
#
# This splits the original train dataset with image and mask of MRI BrainTumor 
#   
# into train and valid  with ratio 0.8 and 0.2
# Origal dataset used here is Brain MRI segmentation:
#
"""
Brain MRI segmentation
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

About Dataset
LGG Segmentation Dataset
Dataset used in:

Mateusz Buda, AshirbaniSaha, Maciej A. Mazurowski "Association of genomic subtypes of 
lower-grade gliomas with shape features automatically extracted by a deep learning 
algorithm." Computers in Biology and Medicine, 2019.
and
Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, 
Katherine B. Peters, Ashirbani Saha "Radiogenomics of lower-grade glioma: 
algorithmically-assessed tumor shape is associated with tumor genomic subtypes 
and patient outcomes in a multi-institutional study with 
The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.

This dataset contains brain MR images together with manual FLAIR abnormality 
segmentation masks.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) 
lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) 
sequence and genomic cluster data available.
Tumor genomic clusters and patient data is provided in data.csv file.
"""

import os
import glob
import shutil
from PIL import Image
import random
import traceback
from skimage.io import imread, imshow, imsave
from skimage.transform import resize

from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize


W = 512
H = 512

# 1 Split original tif files in train_dir into train and valid dataset with ratio 0.8 and 0.2
# 2 Resize the tif files to be 512x512
# 3 Save the resized tif image as jpg file to train/image and valid/image under output_dir 
#    mask tif files as png files to train/mask, and valid/mask

def split_train(train_dir, output_dir="./BrainTumor_master"):
  #train_dir = "./train"
  images_dir = train_dir + "/image/"
  masks_dir  = train_dir + "/mask/"
  image_filepaths = glob.glob(images_dir + "/*.tif")
  num = len(image_filepaths)
  train_num = int(num * 0.8)
  valid_num = int(num * 0.2)
  random.shuffle(image_filepaths)
  train_filepaths = image_filepaths[0: train_num]
  valid_filepaths = image_filepaths[train_num: num]
  
  output_train_dir = output_dir + "/train/"
  output_valid_dir = output_dir + "/valid/"

  save_resize_images(train_filepaths, masks_dir, output_train_dir)

  save_resize_images(valid_filepaths, masks_dir, output_valid_dir)

def convert_test(test_dir, output_dir="./BrainTumor_master"):
  images_dir = test_dir + "/image/"
  masks_dir  = test_dir + "/mask/"
  output_test_dir = output_dir + "/test/"
  image_filepaths = glob.glob(images_dir + "/*.tif")

  save_resize_images(image_filepaths, masks_dir, output_test_dir)


def save_resize_images(train_filepaths, masks_dir, output_train_dir):

  if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)

  output_train_image_dir = output_train_dir + "/image/"
  if not os.path.exists(output_train_image_dir):
    os.makedirs(output_train_image_dir)

  output_train_mask_dir = output_train_dir + "/mask/"
  if not os.path.exists(output_train_mask_dir):
    os.makedirs(output_train_mask_dir)

  for train_filepath in train_filepaths:
     basename = os.path.basename(train_filepath)
     name     = basename.split(".")[0]
     image = Image.open(train_filepath)
     image = image.resize((W, H))
     output_image_file = os.path.join(output_train_image_dir, name + ".jpg")
     print("=== saved jpg file {}".format(output_image_file))

     image.save(output_image_file)

     mask_filepath = os.path.join(masks_dir, name + "_mask.tif")
     print("---mask_filepath {}".format(mask_filepath))
     output_mask_file = os.path.join(output_train_mask_dir, name + ".png")
     save_mask_tif_as_png(mask_filepath,  output_mask_file)
     
     print("=== saved jpg file {}".format(output_mask_file))
     

def save_mask_tif_as_png(tiffile, output_file):

  img = Image.open(tiffile)
  img = img.resize((W, H))

  img_array = np.asarray(img) 

  norm = Normalize(vmin=0, vmax=6)  # ：https://matplotlib.org/users/colormapnorms.html
  converted = cm.Paired(norm(img_array))*255  
  image = Image.fromarray(np.uint8(converted)) 
  image.save(output_file)  


if __name__ == "__main__":
  try:
    input_dir  = "./train"
    output_dir = "./BrainTumor_master"
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   
    split_train(input_dir, output_dir)
    

    input_dir = "./test"
    convert_test(input_dir, output_dir="./BrainTumor_master")

  except:
    traceback.print_exc()

