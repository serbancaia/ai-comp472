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


def evaluate_dataset(dataset_path, model):
    all_predictions = []

    with torch.no_grad():
        for images in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

    return all_predictions


def calculate_confusion_matrix_sum():
    conf_matrix_sum = 0

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            conf_matrix_sum += confusion_matrix[i][j]

    return conf_matrix_sum


def count_class_true_positive(model_class):
    return confusion_matrix[model_class][model_class]


def count_class_false_positive(model_class):
    fp = 0

    for j in range(len(confusion_matrix[model_class])):
        if model_class == j:
            continue
        else:
            fp += confusion_matrix[j][model_class]

    return fp


def count_class_false_negative(model_class):
    fn = 0

    for j in range(len(confusion_matrix[model_class])):
        if model_class == j:
            continue
        else:
            fn += confusion_matrix[model_class][j]

    return fn


def count_class_true_negative(model_class):
    return calculate_confusion_matrix_sum() - (
            count_class_true_positive(model_class) + count_class_false_positive(
        model_class) + count_class_false_negative(
        model_class))


def calculate_class_precision(model_class):
    return count_class_true_positive(model_class) / (
                count_class_true_positive(model_class) + count_class_false_positive(model_class))


def calculate_class_recall(model_class):
    return count_class_true_positive(model_class) / (
                count_class_true_positive(model_class) + count_class_false_negative(model_class))


def calculate_class_f1_measure(model_class):
    return (2 * calculate_class_precision(model_class) * calculate_class_recall(model_class)) / (
            calculate_class_precision(model_class) + calculate_class_recall(model_class))


def calculate_accuracy():
    tp = 0

    for model_class in range(len(confusion_matrix)):
        tp += count_class_true_positive(model_class)

    return tp / calculate_confusion_matrix_sum()


# Declare list of class names and performance metrics table
classes_list = next(os.walk('./GeneratedSplitDataset/test'))[1]
performance_metrics_tabular = [
    ["Model", "Macro-Precision", "Macro-Recall", "Macro-F1", "Micro-Precision", "Micro-Recall", "Micro-F1",
     "Accuracy"], [], [], []]

for model_number in range(3):

    # Declare confusion matrix properties
    confusion_matrix = np.zeros((len(classes_list), len(classes_list)))
    confusion_matrix_tabular = [[], [], [], [], []]

    # Choose which models to evaluate
    while True:
        if model_number == 0:
            chosen_model = input("Choose the main model you would like to evaluate ")
        elif model_number == 1:
            chosen_model = input("Choose the first variant you would like to evaluate ")
        else:
            chosen_model = input("Choose the second variant you would like to evaluate ")
        if os.path.exists(f'./{chosen_model}'):
            # Load the saved model
            if model_number == 0:
                model = main_cnn.ConvNeuralNet()  # Create instance of ConvNeuralNet
            elif model_number == 1:
                model = variant1_cnn.ConvNeuralNet()
            else:
                model = variant2_cnn.ConvNeuralNet()
            model.load_state_dict(torch.load(chosen_model))  # Load the model
            break
        elif os.path.exists(f'./{chosen_model}.pth'):
            # Load the saved model
            if model_number == 0:
                model = main_cnn.ConvNeuralNet()  # Create instance of ConvNeuralNet
            elif model_number == 1:
                model = variant1_cnn.ConvNeuralNet()
            else:
                model = variant2_cnn.ConvNeuralNet()
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

    # Initialize confusion matrix values by make model perform predictions on each test dataset
    for class_int in range(len(classes_list)):
        dataset_path = os.path.join("./GeneratedSplitDataset/test", classes_list[class_int])
        dataset = SingleClassFolder(folder_path=dataset_path, transform=transform)
        dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predicted_class_list = evaluate_dataset(dataset_loader, model)

        for predicted_class in predicted_class_list:
            confusion_matrix[class_int][predicted_class] = confusion_matrix[class_int][predicted_class] + 1

    accuracy = calculate_accuracy()

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
    macro_recall /= len(classes_list)
    macro_f1 /= len(classes_list)

    # Compute the model's micro metrics
    micro_precision = micro_precision_numerator / micro_precision_denominator
    micro_recall = micro_recall_numerator / micro_recall_denominator
    micro_f1 = micro_f1_numerator / micro_f1_denominator

    classes_row = []

    for class_int in range(len(classes_list) + 1):
        if class_int == 0:
            classes_row.append("Class")
        else:
            classes_row.append(classes_list[class_int - 1])

    confusion_matrix_tabular[0] = classes_row

    for class_int in range(1, len(confusion_matrix_tabular)):
        for column in range(len(confusion_matrix_tabular[0])):
            if column == 0:
                confusion_matrix_tabular[class_int].append(classes_list[class_int - 1][column - 1])
            else:
                confusion_matrix_tabular[class_int].append(str(confusion_matrix[class_int - 1][column - 1]))

    print(f"\n{chosen_model}'s Confusion Matrix:")
    for class_int in range(len(classes_list)):
        print(f'{confusion_matrix[class_int]}')

    if model_number == 0:
        performance_metrics_tabular[model_number + 1] = ["Main Model", f'{macro_precision * 100:.4f}%',
                                                         f'{macro_recall * 100:.4f}%', f'{macro_f1 * 100:.4f}%',
                                                         f'{micro_precision * 100:.4f}%', f'{micro_recall * 100:.4f}%',
                                                         f'{micro_f1 * 100:.4f}%', f'{accuracy * 100:.4f}%']
    elif model_number == 1:
        performance_metrics_tabular[model_number + 1] = ["Variant 1", f'{macro_precision * 100:.4f}%',
                                                         f'{macro_recall * 100:.4f}%', f'{macro_f1 * 100:.4f}%',
                                                         f'{micro_precision * 100:.4f}%', f'{micro_recall * 100:.4f}%',
                                                         f'{micro_f1 * 100:.4f}%', f'{accuracy * 100:.4f}%']
    else:
        performance_metrics_tabular[model_number + 1] = ["Variant 2", f'{macro_precision * 100:.4f}%',
                                                         f'{macro_recall * 100:.4f}%', f'{macro_f1 * 100:.4f}%',
                                                         f'{micro_precision * 100:.4f}%', f'{micro_recall * 100:.4f}%',
                                                         f'{micro_f1 * 100:.4f}%', f'{accuracy * 100:.4f}%']

    print(
        f"\n{chosen_model}'s Performance Metrics:\n{performance_metrics_tabular[0]}\n{performance_metrics_tabular[model_number + 1]}\n")

print(f"Complete Performance Metrics:")
for row in performance_metrics_tabular:
    print(f"{row}")
