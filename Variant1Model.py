# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:59:10 2024

@author: Meliimoon
"""

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import os 
from torch.utils.data import DataLoader 

num_epochs = 20
num_classes = 4
learning_rate = 0.0005

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

#Define directories
dataset_dir = './GeneratedSplitDataset'

train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

#Load the datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
validation_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

#Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print("DataLoaders created from split dataset.") #DEBUG

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        #CNN architecture 
        self.conv_layer = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
        )
        
        self.fc_layer = nn.Sequential(
            
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(22*22*16,4),
            
        )
        

    def forward(self, x):
        #Feeding image through convolutional and pooling layers
        x = self.conv_layer(x)
        
        #print('x_shape:',x.shape)
        
        #Flatten
        x = x.view(-1, 22*22*16) #Flatten the tensor to a 1-D vector
        
        #Fully connected layer
        x = self.fc_layer(x)

        return x

model = ConvNeuralNet() #Creating an instance of the CNN

criterion = nn.CrossEntropyLoss() #Includes SoftMax, so we do not need a SoftMax activation function at the end of the last fc layer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)

best_val_loss = float('inf')
patience = 6  # Number of epochs to wait before early stopping
trigger_times = 0

if __name__ == "__main__":
    for epoch in range(num_epochs): 
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images) 
            loss = criterion(outputs, labels)
            
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)  
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_steps, loss.item(),(correct / total) * 100))
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        average_val_loss = val_loss / len(validation_loader)
        accuracy = 100 * correct / total
    
        print(f'Validation Loss: {average_val_loss}, Accuracy: {accuracy}%')
    
        # Early stopping
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            trigger_times = 0
            
            # Saving best-performing model (based on validation set)
            path = './best_model_variation1.pth'
            if os.path.isfile(path): # File exists, will compare best model with current model and will save the better model
                # Define validation evaluation for the saved model
                def current_saved_model_eval(model, dataloader, criterion):
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for images, labels in validation_loader:
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
    
                    average_val_loss = val_loss / len(validation_loader)
                    return average_val_loss
                
                saved_model = ConvNeuralNet() # Model creation as instance of ConvNeuralNet
                saved_model.load_state_dict(torch.load(path)) # Load saved model
                saved_model_loss = current_saved_model_eval(saved_model, validation_loader, criterion) # Evaluate saved model
                
                if average_val_loss < saved_model_loss:  # Compare saved model with current model, save current model as new best model, do nothing otherwise          
                    torch.save(model.state_dict(), 'best_model_variation1.pth') 
                    print("New best model saved.")
                    
            else: # File does not exist, first ever model will be saved
                torch.save(model.state_dict(), 'best_model_variation1.pth') 
                print("First best model saved.")
                
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('Test Accuracy of the model: {} %'.format((correct / total) * 100))