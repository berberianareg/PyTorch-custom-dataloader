#! /bin/bash

# specify root directory where labels.csv and images.csv are stored
root_dir=''
# specify batch size
batch_size=52
# specify the number of pixels to flip on each image to produce noisy image
num_pixel_flips=5
# specify to shuffle 'True' or not to shuffle 'False'
shuffle='False'

# run python script
python Visualize_dataset.py $root_dir $batch_size $num_pixel_flips $shuffle