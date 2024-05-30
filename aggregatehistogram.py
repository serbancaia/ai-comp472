# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:52:16 2024

@author: TristanM2
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#Absolute folder path containing the desired dataset is taken as input and the file name of every image is stored in 'images'
dataset_name = input("What dataset name would you like to plot the aggregate pixel intensity distribution for: ") #Used to title the histogram
folder_path = os.path.join(input("Enter the absolute folder path: ")) 
images = os.listdir(folder_path)
folder_size = len(images)

#List to hold the average pixel intensity of each pixel in the specified dataset
aggregate_pixel_intensity = []

#Initializing the list to size 2304 (the number of pixels in a 48x48 image) in order to ensure that the data types are ints
for i in range(2304):
    aggregate_pixel_intensity.append(0)

#Iterating through every image in the directory and adding the pixel intensity value for each of the 2304 pixels to the aggregate pixel intensity total
for file_name in images:
    new_folder_path = os.path.join(folder_path, file_name)
    img = cv2.imread(new_folder_path, 0)
    img = img.ravel()
    for i in range(len(img)):
        aggregate_pixel_intensity[i] = aggregate_pixel_intensity[i] + img[i]

#Now that each pixel has the total sum of the pixel intensity of every image in the dataset, we must divide each pixel by the number of images in the dataset to get the average intensity for each pixel
for i in range(len(aggregate_pixel_intensity)):
    aggregate_pixel_intensity[i] = aggregate_pixel_intensity[i]//folder_size

#Plotting the histogram and displaying it
plt.hist(aggregate_pixel_intensity, 256, (0, 256))
plt.title("Aggregate Histogram for class: " + dataset_name)
plt.show()