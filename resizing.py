# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:26:05 2024

@author: TristanM2
"""

import os
from PIL import Image

#This script prompts the user to enter an absolute folder path and grayscales and resizes every image in that directory to 48x48, to standardize them

print("STARTING IMAGE RESIZING/GRAYSCALING")

#Absolute folder path containing the dataset is taken as input and the file name of every image is stored in 'images'
folder_path = os.path.join(input("Enter the absolute folder path: "))
images = os.listdir(folder_path)

#Iterate through every image in the dataset, open the image and then resize to 48x48 and convert it to grayscale
for file_name in images:
    new_folder_path = os.path.join(folder_path, file_name)
    print("Resizing and grayscaling: " + new_folder_path)
    img = Image.open(new_folder_path)
    resized_img = img.resize((48,48))
    resized_img = resized_img.convert("L")
    resized_img.save(new_folder_path) 
    
print("IMAGE RESIZING/GRAYSCALING COMPLETED")