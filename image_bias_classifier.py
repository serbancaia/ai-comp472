# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 22:30:27 2024

@author: SerbanCaia
"""
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import shutil
import sys


def choose_from_list(chosen_list: list) -> int:
    # Endlessly take input from user until a valid input is given
    while True:
        # Print each option from the given list with its index value
        for option in range(len(chosen_list)):
            print(f"{option} = {chosen_list[option]}")
        value = input()
        # Wrapped around try-except block in case the given input is not an integer
        try:
            # Alternative case in which user would like to take a break from classifying the images
            if int(value) == -1:
                take_break = input("Would you like to take a break? (yes or no)\nRest assured, your progress will be saved\n")
                if take_break.lower() in ["yes", "ye", "y"]:
                    # Terminate the program if user confirms
                    print("Sounds good. Your progress has been saved")
                    sys.exit()
                else:
                    # Continue asking for input if user did not accept the proposition
                    print("No break then. Moving on...\nPlease choose an integer from the following:")
                    continue
            # Case where user inputs a valid option from the list
            elif int(value) in range(0, len(chosen_list)):
                return int(value)
            # Case where user inputs an invalid integer
            else:
                print("Invalid choice. Please choose another integer from the following:")
        # Handles cases where user inputs a non-numerical value
        except ValueError:
            print("Invalid choice. Please choose another integer from the following:")


# Relative path to the non-bias-classified dataset
dataset_path = './ProjectDatasets'
# Relative path to the bias-classified dataset
biased_dataset_path = './ProjectBiasClassifiedDatasets'

# Lists of classes and bias groups
expressions = ['Angry', 'Focused', 'Happy', 'Neutral']
ages = ['Young', 'Middle-Aged', 'Senior']
genders = ['Male', 'Female', 'Other']
# races = ['Caucasian', 'African-American', 'Asian', 'Middle-Eastern', 'Latino']

# Check to see if the directory tree for the bias-classified dataset was created
if not os.path.isdir(biased_dataset_path):
    print("Directory tree for bias-classified dataset doesn't exist.\nCreating it now.\n")
    # Create the bias-classified dataset directory tree
    os.makedirs(biased_dataset_path)
    for i in expressions:
        current_path = f"{biased_dataset_path}/{i}"
        os.makedirs(current_path)
        
        ages_path = current_path + "/Age"
        os.makedirs(ages_path)
        for i in range(len(ages)):
            os.makedirs(f"{ages_path}/{ages[i]}")

        genders_path = current_path + "/Gender"
        os.makedirs(genders_path)
        for i in range(len(genders)):
            os.makedirs(f"{genders_path}/{genders[i]}")

# Make user decide which dataset class they would like to segment in bias groups
print("Which expression do you want to segment by bias? (must be an integer value)")
expression_int = choose_from_list(expressions)

# Path of the chosen dataset class directory (non-biased-classified dataset)
expression_path = f"{dataset_path}/{expressions[expression_int]}"
# Path of the chosen dataset class directory (biased-classified dataset)
classified_expression_path = f"{biased_dataset_path}/{expressions[expression_int]}"

# Paths of the expression class' bias categories
ages_path = f"{classified_expression_path}/Age"
genders_path = f"{classified_expression_path}/Gender"

# List of images from the chosen dataset class directory
image_list = [f for f in listdir(expression_path) if isfile(join(expression_path, f))]

# Check if the image track configuration file exists;
# if not, create it and initialize it with the class names with each class followed by '0'
if not os.path.exists("./image_track_config.txt"):
    print("Creating image_track_config text file")
    with open("./image_track_config.txt", 'w+') as configFile:
        for i in expressions:
            configFile.writelines(f"{i}\n")
            configFile.writelines('0\n')

# Initialize the variable that will keep track of the image count in the config file for the chosen class
current_image_number = 0

# Read the next image count of given class from the configuration file by iterating through each line
with open("./image_track_config.txt", 'r+') as configFile:
    # Find which line in the config file keeps the image count for the chosen class
    for i in range(0, len(expressions)*2, 2):
        expression = str.strip(configFile.readline())
        if expression == expressions[expression_int]:
            # Update the variables with the line and image count info from the config file
            # and exit the loop once the line is found
            current_image_number = int(str.strip(configFile.readline()))
            break
        else:
            configFile.readline()

# Indicate which image in the chosen class directory the user has reached
if current_image_number == 0:
    print(f"Iterating through the {expressions[expression_int]} dataset for the first time")
else:
    print(f"Currently at image #{current_image_number} in the {expressions[expression_int]} dataset")

# Check to see if the config file has written a higher image count than the actual number of images in the chosen class folder
if current_image_number >= len(image_list):
    # Terminate the program if that is the case
    print(f"You have already reached past the highest index of the {expressions[expression_int]} image list. Terminating program")
    sys.exit()
# Otherwise, proceed with segmenting the images by bias
else:
    # Iterate through all images in the chosen class folder
    for i in range(current_image_number, len(image_list)):

        # Increment the image tracking number of chosen class in the config file by 1 by iterating through each line
        with open("./image_track_config.txt", 'r+') as configFile:
            lines = configFile.readlines()
            for j, line in enumerate(lines):
                if str.strip(line) == expressions[expression_int]:
                    lines[j+1] = str(i) + '\n'
                    break
        with open("./image_track_config.txt", 'w') as configFile:
            configFile.writelines(lines)

        # Display the current image in desktop
        image_path = f"{expression_path}/{image_list[i]}"
        image = Image.open(image_path)
        image.show()

        # Make user decide which bias group the image belongs to for each bias
        print(f"\nWhich age group do you want to assign image {image_list[i]} to? (must be an integer value)")
        age_int = choose_from_list(ages)

        print(f"\nWhich gender group do you want to assign image {image_list[i]} to? (must be an integer value)")
        genders_int = choose_from_list(genders)

        # Copy the image and paste it in each chosen bias group
        shutil.copy(image_path, f"{ages_path}/{ages[age_int]}")
        shutil.copy(image_path, f"{genders_path}/{genders[genders_int]}")

        # Error handling to confirm if image was successfully copied in all folders
        if os.path.exists(f"{ages_path}/{ages[age_int]}/{image_list[i]}") and os.path.exists(f"{genders_path}/{genders[genders_int]}/{image_list[i]}"):
            print(f"\nSuccessfully copied {image_path} to {ages_path}/{ages[age_int]} and to {genders_path}/{genders[genders_int]}")
        else:
            # Terminate the program if image was not found in every chosen bias group folders
            print(f"\nImage copying was unsuccessful. Terminating program")
            sys.exit()

    # Announce the completion of the bias segmentation of the chosen class' image dataset
    print(f"\nYou have successfully segmented all images of the {expressions[expression_int]} class by bias")
