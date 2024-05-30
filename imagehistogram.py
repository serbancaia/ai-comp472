# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:18:42 2024

@author: TristanM2
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

#Absolute folder path containing the dataset is taken as input and the file name of every image is stored in 'images'
folder_path = os.path.join(input("Enter the absolute folder path: "))
images = os.listdir(folder_path)
folder_size = len(images)

#random_images is a list that will contain 15 randomly selected images from a provided dataset
random_images = []
#generated_nums is used to keep track of which indices were generated so we don't display a duplicate histogram
generated_nums = []

#Generate random nums from 0 to the number of images in the dataset and if they weren't used before, then add the image at the index of that random num to the list
while True:
    random_num = random.randrange(folder_size)
    if random_num not in generated_nums:
        generated_nums.append(random_num)
        random_images.append(images[random_num])
    #Break once 15 images have been randomly selected
    if len(generated_nums) == 15:
        break

#Iterate through the 15 randomly selected images in the dataset and flatten the image to a 1D array in order to plot the histogram
for file_name in random_images:
    new_folder_path = os.path.join(folder_path, file_name)
    img = cv2.imread(new_folder_path, 0) #Open the randomly selected image
    plt.hist(img.ravel(), 256, (0, 256))
    plt.title("Histogram for image: " + file_name)
    plt.show()