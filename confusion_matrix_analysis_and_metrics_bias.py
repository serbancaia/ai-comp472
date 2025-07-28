# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:23:41 2024

@author: SerbanCaia, Meliimoon
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import MainCNNModel as main_cnn
import Variant1Model as variant1_cnn
import Variant2Model as variant2_cnn
import os
from torch.utils.data import Dataset
import numpy as np


# Entire dataset
# Custom dataset for images in a single folder
class SingleClassFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.endswith(('png', 'jpg'))]  # creates a list of the paths of all .png and .jpg image files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image


# Evaluate dataset with a given model
def evaluate_dataset(dataset_path, model):
    all_predictions = []

    with torch.no_grad():
        for images in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

    return all_predictions


# Calculate the sum of the confusion matrix
def calculate_confusion_matrix_sum():
    conf_matrix_sum = 0

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            conf_matrix_sum += confusion_matrix[i][j]

    return conf_matrix_sum


# Count true positives for a given class
def count_class_true_positive(model_class):
    return confusion_matrix[model_class][model_class]


# Count false positives for a given class
def count_class_false_positive(model_class):
    fp = 0

    for j in range(len(confusion_matrix[model_class])):
        if model_class == j:
            continue
        else:
            fp += confusion_matrix[j][model_class]

    return fp


# Count false negatives for a given class
def count_class_false_negative(model_class):
    fn = 0

    for j in range(len(confusion_matrix[model_class])):
        if model_class == j:
            continue
        else:
            fn += confusion_matrix[model_class][j]

    return fn


# Count true negatives for a given class
def count_class_true_negative(model_class):
    return calculate_confusion_matrix_sum() - (
            count_class_true_positive(model_class) + count_class_false_positive(
        model_class) + count_class_false_negative(
        model_class))


# Calculate precision for a given class
def calculate_class_precision(model_class):
    return count_class_true_positive(model_class) / (
                count_class_true_positive(model_class) + count_class_false_positive(model_class))


# Calculate recall for a given class
def calculate_class_recall(model_class):
    return count_class_true_positive(model_class) / (
                count_class_true_positive(model_class) + count_class_false_negative(model_class))


# Calculate F1 score for a given class
def calculate_class_f1_measure(model_class):
    return (2 * calculate_class_precision(model_class) * calculate_class_recall(model_class)) / (
            calculate_class_precision(model_class) + calculate_class_recall(model_class))


# Calculate overall accuracy
def calculate_accuracy():
    tp = 0

    for model_class in range(len(confusion_matrix)):
        tp += count_class_true_positive(model_class)

    return tp / calculate_confusion_matrix_sum()


# Declare list of class names and performance metrics table
bias_classes_list = next(os.walk('./GeneratedSplitBiasDataset/test'))[1]
classes_list = ['Angry', 'Focused', 'Happy', 'Neutral']
performance_metrics_tabular = [
    ["Group", "#Images", "Accuracy", "Precision", "Recall", "F1"], [], [], [], [], [], [], [], [], []]

# Choose which model to evaluate
while True:
    chosen_model = input("Choose the main model you would like to evaluate ")
    if os.path.exists(f'./{chosen_model}'):
        # Load the saved model
        model = main_cnn.ConvNeuralNet()  # Create instance of ConvNeuralNet
        model.load_state_dict(torch.load(chosen_model))  # Load the model
        break
    elif os.path.exists(f'./{chosen_model}.pth'):
        # Load the saved model
        model = main_cnn.ConvNeuralNet()  # Create instance of ConvNeuralNet
        model.load_state_dict(torch.load(chosen_model + '.pth'))  # Load the model
        break
    else:
        print("Model doesn't exist. Please choose another one\n")

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

total_image_number = 0
current_perf_row = 0
overall_accuracy_numerator = 0
overall_precision_numerator = 0
overall_recall_numerator = 0
overall_f1_numerator = 0
total_subgroup_number = 0

for bias_class in bias_classes_list:
    image_number_per_bias_class = 0
    precision_numerator = 0
    recall_numerator = 0
    f1_numerator = 0
    accuracy_numerator = 0
    total_subgroup_number += len(next(os.walk(f'./GeneratedSplitBiasDataset/test/{bias_class}'))[1])
    bias_subgroups = next(os.walk(f'./GeneratedSplitBiasDataset/test/{bias_class}'))[1]

    for bias_subgroup in bias_subgroups:
        # Declare confusion matrix properties
        confusion_matrix = np.zeros((len(classes_list), len(classes_list)))
        confusion_matrix_tabular = [[], [], [], [], []]
        current_perf_row += 1
        bias_subgroup_classes = next(os.walk(f'./GeneratedSplitBiasDataset/test/{bias_class}/{bias_subgroup}'))[1]

        # Initialize confusion matrix values by making model perform predictions on each test dataset
        for class_int in range(len(classes_list)):
            dataset_path = os.path.join(f"./GeneratedSplitBiasDataset/test/{bias_class}/{bias_subgroup}/{bias_subgroup_classes[class_int]}")
            dataset = SingleClassFolder(folder_path=dataset_path, transform=transform)
            dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            predicted_class_list = evaluate_dataset(dataset_loader, model)

            for predicted_class in predicted_class_list:
                confusion_matrix[class_int][predicted_class] = confusion_matrix[class_int][predicted_class] + 1

        image_number_per_bias_class += calculate_confusion_matrix_sum()
        total_image_number += calculate_confusion_matrix_sum()

        # Calculate performance metrics
        accuracy = calculate_accuracy()
        accuracy_numerator += accuracy
        overall_accuracy_numerator += accuracy

        macro_precision = 0
        micro_precision_numerator = 0
        micro_precision_denominator = 0

        macro_recall = 0
        micro_recall_numerator = 0
        micro_recall_denominator = 0

        macro_f1 = 0
        micro_f1_numerator = 0
        micro_f1_denominator = 0

        for class_int in range(len(classes_list)):
            # Add the current class' precision, recall, and f1-measure to the macro-precision, macro-recall,
            # and macro-f1-measure variables respectively
            macro_precision += calculate_class_precision(class_int)
            macro_recall += calculate_class_recall(class_int)
            macro_f1 += calculate_class_f1_measure(class_int)

            # Add the current class' true positive score to the numerator variables
            micro_precision_numerator += count_class_true_positive(class_int)
            micro_recall_numerator += count_class_true_positive(class_int)
            micro_f1_numerator += count_class_true_positive(class_int)

            # Add the current class' tp+fp, tp+fn, and tp+(1/2)(fp + fn) to the respective denominator variable
            micro_precision_denominator += count_class_true_positive(class_int) + count_class_false_positive(class_int)
            micro_recall_denominator += count_class_true_positive(class_int) + count_class_false_negative(class_int)
            micro_f1_denominator += count_class_true_positive(class_int) + (1 / 2) * (
                        count_class_false_positive(class_int) + count_class_false_negative(class_int))

        # Compute the model's macro metrics
        macro_precision /= len(classes_list)
        precision_numerator += macro_precision
        overall_precision_numerator += macro_precision

        macro_recall /= len(classes_list)
        recall_numerator += macro_recall
        overall_recall_numerator += macro_recall

        macro_f1 /= len(classes_list)
        f1_numerator += macro_f1
        overall_f1_numerator += macro_f1

        # Compute the model's micro metrics
        micro_precision = micro_precision_numerator / micro_precision_denominator
        micro_recall = micro_recall_numerator / micro_recall_denominator
        micro_f1 = micro_f1_numerator / micro_f1_denominator

        classes_row = []

        # Fill confusion matrix table with headers
        for class_int in range(len(classes_list) + 1):
            if class_int == 0:
                classes_row.append("Class")
            else:
                classes_row.append(classes_list[class_int - 1])

        confusion_matrix_tabular[0] = classes_row

        # Fill confusion matrix table with performance metrics
        for class_int in range(1, len(confusion_matrix_tabular)):
            for column in range(len(confusion_matrix_tabular[0])):
                if column == 0:
                    confusion_matrix_tabular[class_int].append(classes_list[class_int - 1][column - 1])
                else:
                    confusion_matrix_tabular[class_int].append(str(confusion_matrix[class_int - 1][column - 1]))

        # Display the current model's confusion matrix and performance metrics
        print(f"\n{bias_subgroup}'s Confusion Matrix:")
        print(f"{['Actual/Predicted', 'Angry', 'Focused', 'Happy', 'Neutral']}")
        for class_int in range(len(classes_list)):
            print(f'{classes_list[class_int]}\t\t\t{confusion_matrix[class_int]}')

        performance_metrics_tabular[current_perf_row] = [f"{bias_subgroup}", f'{calculate_confusion_matrix_sum():.0f}', f'{accuracy * 100:.4f}%', f'{macro_precision * 100:.4f}%',
                                                             f'{macro_recall * 100:.4f}%', f'{macro_f1 * 100:.4f}%']
        print(
            f"\n{bias_subgroup}'s Performance Metrics:\n{performance_metrics_tabular[0]}\n{performance_metrics_tabular[current_perf_row]}\n")

    current_perf_row += 1
    performance_metrics_tabular[current_perf_row] = ["Total/Average", f'{image_number_per_bias_class:.0f}', f'{(accuracy_numerator/len(bias_subgroups)) * 100:.4f}%', f'{(precision_numerator/len(bias_subgroups)) * 100:.4f}%',
                                                             f'{(recall_numerator/len(bias_subgroups)) * 100:.4f}%', f'{(f1_numerator/len(bias_subgroups)) * 100:.4f}%']

current_perf_row += 1
performance_metrics_tabular[current_perf_row] = ["Overall System Total/Average", f'{total_image_number:.0f}', f'{(overall_accuracy_numerator/total_subgroup_number) * 100:.4f}%', f'{(overall_precision_numerator/total_subgroup_number) * 100:.4f}%',
                                                             f'{(overall_recall_numerator/total_subgroup_number) * 100:.4f}%', f'{(overall_f1_numerator/total_subgroup_number) * 100:.4f}%']

# Display model performance metrics
print(f"\nComplete Performance Metrics:")
for row in performance_metrics_tabular:
    print(f"{row}")
