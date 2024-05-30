# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:08 2024

@authors: TristanM2, serbancaia
"""

import cv2
import os

#Get the absolute path to the directory that you wish to check for duplicate images
folder_path = os.path.join(input("Enter the absolute folder path to verify for duplicates: "))
images = os.listdir(folder_path)

#If any dupe is found, found_dupe will be set to True, and an appropriate print statement will displayed
found_dupe = False 
#ctr is used to keep track of the number of duplicate images, used when printing how many dupes were detected
ctr = 0

#Iterate through the images in the provided directory and compare each image with every other image in the directory (nb of iterations performed: n*(n-1)/2)
#The comparison of images is done by comparing the pixel intensity value at each of the 2304 indices of the flattened 48x48 images
for i in range(len(images)):
    file_name = images[i]
    new_folder_path = os.path.join(folder_path, file_name)
    if os.path.exists(new_folder_path):
        img = cv2.imread(new_folder_path, 0)
        img = img.ravel() #Flatten the 48x48 array of pixels to a 1x2304 1-D array

        for j in range(i + 1, len(images)):
            dupe_file = images[j]
            if os.path.exists(dupe_folder_path):
                dupeimg = cv2.imread(dupe_folder_path, 0)
                dupeimg = dupeimg.ravel()

                for i in range(len(img)):
                    if img[i] != dupeimg[i]: #If one image's pixel intensity is different at a given index, then it is not the same image, so break the loop and check the next image
                        break;
                    elif i == (len(img)-1): #If the loop did NOT break, and it reached the last pixel, then every single pixel between the 2 images was the exact same, thus they were duplicates
                        os.remove(dupe_folder_path) #Delete the duplicate image
                        print("The file " + dupe_file + " is a duplicate image of " + file_name + " and has been deleted")
                        found_dupe = True
                        ctr = ctr + 1
           
if not found_dupe:
    print("No duplicate images were found!")
else: 
    print()
    print("There were " + str(ctr) + " duplicate images. We deleted them for you!")
