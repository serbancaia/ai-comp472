# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:53:11 2024

@author: TristanM2
"""
import matplotlib.pyplot as plt
import numpy as np
import os

#Absolute folder path containing the dataset is taken as input and the file name of every image is stored in '_images'
happy_folder_path = os.path.join(input("Enter the absolute folder path for HAPPY: "))
happy_images = os.listdir(happy_folder_path)
happy_folder_size = len(happy_images)

angry_folder_path = os.path.join(input("Enter the absolute folder path for ANGRY: "))
angry_images = os.listdir(angry_folder_path)
angry_folder_size = len(angry_images)

neutral_folder_path = os.path.join(input("Enter the absolute folder path for NEUTRAL: "))
neutral_images = os.listdir(neutral_folder_path)
neutral_folder_size = len(neutral_images)

focused_folder_path = os.path.join(input("Enter the absolute folder path for FOCUSED: "))
focused_images = os.listdir(focused_folder_path)
focused_folder_size = len(focused_images)

#x-axis of bar graph containing the dataset classes
x = np.array(["HAPPY", "ANGRY", "NEUTRAL", "FOCUSED"]) 

#y-axis of bar graph containing the size of each dataset class
y = np.array([happy_folder_size, angry_folder_size, neutral_folder_size, focused_folder_size]) 

#Plotting and displaying the bar graph for the number of images in each dataset class
plt.bar(x,y)
plt.show()
