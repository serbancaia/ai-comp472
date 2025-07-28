# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:40:42 2024

@author: Meliimoon
"""

import torchvision.transforms as transforms
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

#Relative path to the dataset
dataset_path = './ProjectDatasets'
#Output folder of split datasets
split_dir = './GeneratedSplitDataset'

#Create the dataset object
combined_dataset = ImageFolder(dataset_path, transform)

#Get class-to-index dictionary mapping
class_to_idx = combined_dataset.class_to_idx #Dictionary of class and index -> {'Angry': 0, 'Focused': 1, 'Happy': 2, 'Neutral': 3}
idx_to_class = {v: k for k, v in class_to_idx.items()} #Dictionary of index and class -> {0: 'Angry', 1: 'Focused', 2: 'Happy', 3: 'Neutral'}

#Create directories for the splits
for split in ['train', 'validation', 'test']:
    for class_name in class_to_idx.keys():
        os.makedirs(os.path.join(split_dir, split, class_name), exist_ok=True)

#Extract all file paths and labels
file_paths = [s[0] for s in combined_dataset.samples]
labels = [s[1] for s in combined_dataset.samples]

#Split the dataset into train/validation/test portion of the dataset by splitting 70/15/15 for train/validation/test
train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, stratify=labels, random_state=42) #random_state sets a random num generation seed for ensuring the same dataset split is generated each time (42 is arbitrary)
validation_paths, test_paths, validation_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

#Copy image files to their respective folders in the output folder
def copy_files(file_paths, target_dir):
    for file_path in file_paths:
        shutil.copy(file_path, os.path.join(target_dir, os.path.basename(file_path)))

# Copy image files to their respective directories
for file_path, label in zip(train_paths, train_labels): #train_paths = list of image file names, train_labels = list of expression class values (ex: 0 = Angry class)
    copy_files([file_path], os.path.join(split_dir, 'train', idx_to_class[label])) 

for file_path, label in zip(validation_paths, validation_labels):
    copy_files([file_path], os.path.join(split_dir, 'validation', idx_to_class[label]))

for file_path, label in zip(test_paths, test_labels):
    copy_files([file_path], os.path.join(split_dir, 'test', idx_to_class[label]))

print("Dataset has been split and images are copied to respective directories.")

