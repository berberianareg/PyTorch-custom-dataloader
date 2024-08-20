"""Visualize custom dataset by leveraging PyTorch dataset APIs.

Notes
-----
  This script is version v0. It provides the base for all subsequent
  iterations of the project.
  
Requirements
------------
  See "requirements.txt"
  
Comments
--------
  Custom dataset contains 52 images in total
    26 lowercase letters
    26 uppercase letters
  Each image has bipolar encoding scheme (-1 and 1)
  Each image is of shape 7x7 (49 dimensions)
  Each image has corresponding label (0-25)
  
"""
# %% import libraries and modules
from Build_dataset import LettersDataset
from Build_dataset import ToTensor, Flatten, Binarize, PixelFlip
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# %% argpars
parser = argparse.ArgumentParser(prog='Visualize_dataset.py', description='PyTorch custom dataloader')
parser.add_argument('root_dir', type=str, help='specify root directory where labels.csv and images.csv are stored')
parser.add_argument('batch_size', type=int, choices=range(1, 53), help='specify batch size')
parser.add_argument('num_pixel_flips', type=int, choices=range(0, 50), help='specify the number of pixels to flip on each image to produce noisy image')
parser.add_argument('shuffle', type=str, choices=('True', 'False'), help='specify to shuffle (True) or not to shuffle (False) batched images')

args = parser.parse_args()

if args.shuffle == 'True':
  args.shuffle = True
elif args.shuffle == 'False':
  args.shuffle = False

print(args)
# %% load original dataset
labels_file=os.path.join(args.root_dir, 'labels.csv')
images_file=os.path.join(args.root_dir, 'images.csv')

letters_dataset = LettersDataset(
    labels_file=labels_file,
    images_file=images_file,
    transform=transforms.Compose([ToTensor(), Flatten(), Binarize()])
    )

# # iterate through the original data samples
# for index, sample in enumerate(letters_dataset):
#   print(sample['image_name'], sample['image'].shape, sample['label'])

# load minibatch of original images
dataloader = DataLoader(letters_dataset, batch_size=args.batch_size, sampler=None, shuffle=args.shuffle, num_workers=0)
# get minibatch of original images
batched_data = next(iter(dataloader))

# get image names within entire dataset
image_names = letters_dataset.labels_file.index

# get image names within minibatch
batched_image_names = batched_data['image_name']
# get index of each image found in minibatch - this will later be leveraged to preserve data loading order on noisy images
sampler = [np.where(batched_image_name == image_names)[0][0] for batched_image_name in batched_image_names if batched_image_name in image_names]

# %% load noisy dataset (with pixel flip noise)
letters_dataset_noisy = LettersDataset(
    labels_file=labels_file,
    images_file=images_file,
    transform=transforms.Compose([ToTensor(), Flatten(), Binarize(), PixelFlip(args.num_pixel_flips)])
    )

# load minibatch of noisy images
dataloader_noisy = DataLoader(letters_dataset_noisy, batch_size=args.batch_size, sampler=sampler, num_workers=0)
# get minibatch of noisy images
batched_noisy_data = next(iter(dataloader_noisy))

# %% make grids
padding = 1 # amount of padding
nrow = 13 # number of images displayed in each row of the grid

# original images grid
grid = utils.make_grid(batched_data['image'].reshape(args.batch_size, 7, 7)[:, None, :, :], padding=padding, nrow=nrow)[0, :, :]
# noisy images grid
grid_noisy = utils.make_grid(batched_noisy_data['image'].reshape(args.batch_size, 7, 7)[:, None, :, :], padding=padding, nrow=nrow)[0, :, :]

# %% plot results
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].imshow(grid)
ax[0].set_title('original images')
ax[0].axis('off')
ax[1].imshow(grid_noisy)
ax[1].set_title('noisy images')
ax[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(args.root_dir, 'figure.png'))
plt.close()