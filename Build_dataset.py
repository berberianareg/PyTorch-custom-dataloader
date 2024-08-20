"""Build custom dataset by leveraging PyTorch dataset APIs.

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
from torch.utils.data import Dataset
import pandas as pd
import torch

# %% build LettersDataset class
class LettersDataset(Dataset):
  """
  Arguments:
    labels_file (string): Path to the csv file with labels
    images_file (string): Path to the csv file with images
    transform (callable, optional): Optional transform to be applied on a sample
  """
  def __init__(self, labels_file, images_file, transform=None):
    self.labels_file = pd.read_csv(labels_file, index_col='image_name')
    self.images_file = pd.read_csv(images_file, index_col='image_name')
    self.transform = transform

  def __len__(self):
    return len(self.labels_file)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    image_name = self.images_file.index[idx]
    image = self.images_file.iloc[idx].values.reshape(7, 7, 1) # H x W x C
    label = self.labels_file.iloc[idx].values

    sample = {'image_name': image_name,
              'image': image,
              'label': label}

    if self.transform:
      sample = self.transform(sample)
    
    return sample

# %% implement Transforms
class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""
  def __call__(self, sample):
    image_name, image, label = sample['image_name'], sample['image'], sample['label']
    # swap axes because
    # numpy image: H x W x C
    # torch image: C x H x W
    image = image.transpose((2, 0, 1))

    return {'image_name': image_name,
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)}

class Flatten(object):
  """Flatten sample image."""
  def __call__(self, sample):
    image_name, image, label = sample['image_name'], sample['image'], sample['label']
    image = image.flatten() # flatten image

    return {'image_name': image_name,
            'image': image,
            'label': label}
  
class Binarize(object):
  """Binarize sample image from bipolar (1 and -1) to binary (1 and 0)."""
  def __call__(self, sample):
      image_name, image, label = sample['image_name'], sample['image'], sample['label']
      image = (image + 1) / 2 # binarize image

      return {'image_name': image_name,
              'image': image,
              'label': label}
  
class PixelFlip(object):
  """Perform pixel flip on sample image.
  Arguments:
    num_pixel_flips (int): Desired number of pixel flips on image
  """
  def __init__(self, num_pixel_flips):
    self.num_pixel_flips = num_pixel_flips

  def __call__(self, sample):
    image_name, image, label = sample['image_name'], sample['image'], sample['label']
    num_units = len(image)
    # select pixel flip indices at random
    flip_indices = torch.randperm(num_units)[:self.num_pixel_flips]
    # clone original image
    noisy_image = torch.clone(image)
    if -1 in image: # if image is bipolar (-1 and 1)
      # flip pixels at specified indices (-1 -> 1 and 1 -> -1)
      noisy_image[flip_indices] = noisy_image[flip_indices] * -1
    elif 0 in image: # if image is binary (0 and 1)
      # flip pixels at specified indices (0 -> 1 and 1 -> 0)
      noisy_image[flip_indices] = abs(noisy_image[flip_indices] - 1)

    return {'image_name': image_name,
            'image': noisy_image,
            'label': label}
    