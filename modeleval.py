# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:46:07 2024

@author: Meliimoon
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from MainCNNModel import ConvNeuralNet
import os
from collections import Counter
from torch.utils.data import Dataset

# Load the saved model
model = ConvNeuralNet() # Create instance of ConvNeuralNet
model.load_state_dict(torch.load('main_best_model.pth'))  # Load the best-performing model --> main_best_model.pth is the final model we saved from our MainCNNModel_callable.py architecture that we conducted evaluations on
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def evaluate_single_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0) #adds extra dimension at index 0 to result in [batch size, number of channels, height, width] -> results in [1, 1, 48, 48]
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

print("*NOTE*\n \t'0' indicates 'Angry' \n \t'1' indicates 'Focused' \n \t'2' indicates 'Happy' \n \t'3' indicates 'Neutral' \n")
eval_choice = input("Enter 1 to evaluate an individual image from the dataset \nEnter 2 to classify a complete dataset \nEnter 3 to classify a custom image (Application Mode):\n   > ")
print()

if eval_choice == '1': #Single image evaluation
    # Single image
    print('Single image evaluation:')
    expression_class = input("Enter the class you want to evaluate: ")
    image_name = input("Enter the image file name (including the file type) you want to evaluate: ")
    image_path = os.path.join("./GeneratedSplitDataset/test", expression_class, image_name)
    
    print("Expected class for the individual image: ", expression_class)
    
    predicted_class = evaluate_single_image(image_path, model)
    if predicted_class == 0:
        print('Predicted class for the individual image: Angry')
    elif predicted_class == 1:
        print('Predicted class for the individual image: Focused')
    elif predicted_class == 2:
        print('Predicted class for the individual image: Happy')
    elif predicted_class == 3:
        print('Predicted class for the individual image: Neutral')

elif eval_choice == '2': #Dataset evaluation
    # Entire dataset
    # Custom dataset for images in a single folder
    class SingleClassFolder(Dataset):
        def __init__(self, folder_path, transform=None):
            self.folder_path = folder_path
            self.transform = transform
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg'))] #creates a list of the paths of all .png and .jpg image files
    
        def __len__(self):
            return len(self.image_paths)
    
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
    
    print('Entire Dataset evaluation:')
    dataset_class = input("Enter the class you want to evaluate: ")
    dataset_path = os.path.join("./GeneratedSplitDataset/test", dataset_class)
    dataset = SingleClassFolder(folder_path=dataset_path, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    def evaluate_dataset(dataset_path, model):
        all_predictions = []
    
        with torch.no_grad():
            for images in dataset_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
    
        return all_predictions
    
    predicted_class = evaluate_dataset(dataset_loader, model)
    
    print("Expected class for the dataset: ", dataset_class)
    # Calculate the most frequently predicted class
    predicted_class_counts = Counter(predicted_class)
    print('Model prediction counter: ', predicted_class_counts)
    most_frequent_class = predicted_class_counts.most_common(1)[0][0] #.most_common(1) returns [(most_common, count)], [0] returns (most_common, count), [0] returns most_common
    
    if most_frequent_class == 0:
        print('Predicted class for the dataset: Angry')
    elif most_frequent_class == 1:
        print('Predicted class for the dataset: Focused')
    elif most_frequent_class == 2:
        print('Predicted class for the dataset: Happy')
    elif most_frequent_class == 3:
        print('Predicted class for the dataset: Neutral')
        
elif eval_choice == '3': #Custom image evaluation
    #Custom image
    print('Custom image evaluation:')
    directory_path = input("Enter the absolute path of the custom image that you want to classify: ")  
    
    predicted_class = evaluate_single_image(directory_path, model)
    if predicted_class == 0:
        print('Predicted class for the custom image: Angry')
    elif predicted_class == 1:
        print('Predicted class for the custom image: Focused')
    elif predicted_class == 2:
        print('Predicted class for the custom image: Happy')
    elif predicted_class == 3:
        print('Predicted class for the custom image: Neutral')