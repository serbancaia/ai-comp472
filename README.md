------------------------------------------------------------------------------------------------------
# Group name: FS-4  
# Group members: Tristan Manni-Mirolla (40112168), Alin Caia (40246955), Melissa Ananian (40112159)
------------------------------------------------------------------------------------------------------

The following packages must be installed in your environment to execute the Python scripts that we wrote for this project:   

	>pip install pillow  
	>conda install scikit-learn  
	>pip install matplotlib OR conda install matplotlib  
	>pip install opencv-python==4.5.3.56  
 	>conda install pytorch torchvision -c pytorch

For OpenCV, using the latest version was not compatible with Python 3.6, so we installed a previous version that was found to work online.  
All the Python scripts prompt the user to enter an absolute file path to a dataset in the format 'C:\Users\Tristan\Documents\Concordia\COMP472\Project-Part1'.  

# Project Part 1

To execute our Python scripts on our dataset, you must first download our dataset from our GitHub Repository.  
Direct link: https://github.com/serbancaia/ai-comp472/blob/ce8d08fd58db7d4b2cab80755f3ebc5cf968d160/ProjectDatasets.zip  
<b>*NOTE: This dataset found in branch "phase1" of our respository is NOT used for Part 2 of the project and is now obsolete</b>

## Our '.py' Python script files:  
### ---STEPS FOR EXECUTING OUR CODE FOR (a) DATA CLEANING AND (b) DATA VISUALIZATION---  
Step 0) Open the Python environment with the above packages installed through the command prompt and navigate to the directory containing the python scripts.  
<br />
	a)  
	-dupecheck.py: This script is used to check for duplicate images within a directory by comparing every pixel's intensity and deletes duplicates 
 
		Step 1) Execute >python dupecheck.py  
		Step 2) Enter the absolute folder path to the dataset that you would like to check for duplicates (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Happy")  
		Step 3) Wait for the script to run. Any duplicate images will be identified (and printed) and deleted. The script may take up to ~45 seconds for a dataset consisting of 500 images  
<br />
	-resizing.py: This script grayscales and resizes all images in a directory to 48x48px, necessary to standardize our datasets  

		Step 1) Execute >python resizing.py  
		Step 2) Enter the absolute folder path to the dataset that you would like to resize to 48x48px and grayscale (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Happy")  
		Step 3) The script will grayscale and resize every image in the directory automatically  
		**NOTE: THIS SCRIPT SHOULD BE RUN ON A DATASET BEFORE RUNNING ANY OTHER PYTHON SCRIPT IN ORDER TO STANDARDIZE THE DATASETS!  
<br />
	b) <br />      
	-aggregatehistogram.py: This script is used to plot/display the aggregated pixel intensity distribution for an entire dataset class  

		Step 1) Execute >python aggregatehistogram.py  
		Step 2) Provide the name of the dataset that you would like to plot the aggregated pixel intensity distribution for (ex: "Happy")  
		Step 3) Enter the absolute folder path to the dataset (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Happy")  
		Step 4) The aggregated pixel intensity distribution histogram for the entire dataset class is displayed  
			The x-axis of the histogram denotes the pixel intensity from 0-255 (0 being black and 255 being white)  
			The y-axis of the histogram denotes the number of pixels that have that pixel intensity value  
<br />
	-imagebar.py: This script is used to plot/display the bar graph showing the number of images in each dataset class. Takes 4 dataset absolute folder paths as input  

		Step 1) Execute >python imagebar.py  
		Step 2) Enter the absolute folder path for the "Happy" dataset (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Happy")  
		Step 3) Enter the absolute folder path for the "Angry" dataset (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Angry")  
		Step 4) Enter the absolute folder path for the "Neutral" dataset (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Neutral")  
		Step 5) Enter the absolute folder path for the "Focused" dataset (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Focused")  
		Step 6) The bar graph showing the number of images in each dataset is displayed  
			The x-axis of the bar graph denotes the dataset name  
			The y-axis of the bar graph denotes the number of images  
<br />
	-imagehistogram.py: This script randomly selects 15 images in a provided directory and plots/displays those images' pixel intensity histograms  

		Step 1) Execute >python imagehistogram.py  
		Step 2) Enter the absolute folder path to the dataset that you would like to generate the 15 randomly selected pixel intensity histograms for (ex: "C:\Users\Tristan\Desktop\Concordia\COMP 472\Project-Part1\Training\Happy")  
		Step 3) 15 pixel intensity histograms corresponding to the 15 randomly selected images will be displayed one at a time (close the histogram for the next one to be displayed)  
			The x-axis of the histogram denotes the the pixel intensity from 0-255 (0 being black and 255 being white)  
			The y-axis of the histogram denotes the number of pixels that have that pixel intensity value

## Dataset.pdf:  
<br />
	This file contains the provenance of the datasets and images used for our project's dataset. Referencing of the online dataset and
	their licensing type is listed. This document also showcases the custom images our team created for each class for our dataset along
	with the necessary modifications for standardization with the rest of our dataset. Lastly, this file contains a sample set of 25
	representative images for each class in our dataset.  
 <br />
	A direct link to our dataset's .zip file (uploaded to our GitHub repository) is provided at the top of this file.  
 <br />
	The purpose of this file is to display information about our dataset and the online datasets we used as well as showcase some of the
	images that could be found in our dataset, as we are unable to upload the entire dataset to Moodle due to its large size.  

## ProjectReport-Part1.pdf:  
<br />
	This file contains the complete report of our work for Part 1 of our project.
 <br />
 	The report includes a title page, information about the creation of our dataset, the explanation of our data cleaning techniques and challenges, the explanation of the labeling methods and 
	challenges, and the visualizations of our dataset. These visualizations include a bar graph to display the number of images our dataset 
	contains in each expression's training data class, one aggregrated pixel intensity distribution histogram for each expression's training 
	data class, and pixel intensity histograms for 15 randomly chosen images in each expression's training data class. Lastly, this report 
	contains a reference section for all the material used/referenced throughout this deliverable.  

## Expectation of Originality files:  
<br />
	Expectations-of-Originality-May30-2024-AlinCaia.pdf: <br />
		This is the Expectation of Originality form signed by Alin Caia <br />
<br />
	Expectations-of-Originality-MelissaAnanian40112159.pdf: <br />
		This is the Expectation of Originality form signed by Melissa Ananian <br />
<br />
	Expectations-of-Originality-TristanManniMirolla40112168.pdf: <br />
		This is the Expectation of Originality form signed by Tristan Manni-Mirolla <br />
<br />
	These forms are provided to attest to the originality of our work.  

# Project Part 2
Updates on Dataset folders found in 'phase2' branch of our repository:
<br />
>GeneratedSplitDataset.zip: https://github.com/serbancaia/ai-comp472/blob/phase2/GeneratedSplitDataset.zip <br />
This is the main dataset we used for the training and testing of our models. It was automatically generated by the datasetsplitter.py script
<br /><br />
ProjectDatasets.zip: https://github.com/serbancaia/ai-comp472/blob/phase2/ProjectDatasets.zip <br />
This is the dataset we used in the datasetsplitter.py script to automatically divide our dataset's classes into 'train', 'validation', and 'test' folders. This dataset is different from the one found in the 'phase1' branch of our repository, as it does not have the manual division of 'train' and 'test' folders for the classes
<br />

<b>You may find and download the best models we saved for each CNN model in a Google Drive. These are the saved models we used to test, evaluate, and analyze our Main (main_best_model.pth), Variant 1 (variant1.pth), and Variant 2 (variant2.pth) models: https://drive.google.com/drive/folders/18PRlHVx4nhuY-eYKxq6JLFAbrbXTOvpV?fbclid=IwZXh0bgNhZW0CMTAAAR1bsVIxtD6kl9RPZ9i2Cq6TWziS67YKXLrfkwSkCTp14POpHusXiqG1Ujc_aem_ZmFrZWR1bW15MTZieXRlcw </b>
<br />
NOTE: Our MainCNNModel.py, Variant1Model.py, and Variant2Model.py files may override our saved best model .pth files if a better model is found when executing the scripts. If this occurs, redownload the original .pth files from our Google Drive to use for evaluation in the modeleval.py and confusion_matrix_analysis_and_metrics.py files to generate the same numbers found in the Evaluation section of our report.
<br /><br />

## Our '.py' Python script files:  
### ---STEPS FOR EXECUTING OUR CODE FOR (a) CNN MAIN MODEL AND VARIANTS (b) AUTOMATIC DATASET SPLITTING (c) LOADING, EVALUATION, AND TESTING AND (d) SUPPLEMENTARY FILES---
Step 0) Open the Python environment with the above packages installed through the command prompt and navigate to the directory containing the python scripts.  
<br />
	a)  
	-MainCNNModel.py: This script is our Main CNN model's architecture and training file. The model with the lowest validation loss gets saved as the best model as 'main_best_model.pth' 
 
		Step 1) Execute >python MainCNNModel.py  
		
<br />
  -Variant1Model.py: This script is our Variant 1 CNN model's architecture and training file. The model with the lowest validation loss gets saved as the best model as 'variant1.pth' 

		Step 1) Execute >python Variant1Model.py  
  
<br />
  -Variant2Model.py: This script is our Variant 2 CNN model's architecture and training file. The model with the lowest validation loss gets saved as the best model as 'variant2.pth' 

		Step 1) Execute >python Variant2Model.py    
b) <br />      
	-datasetsplitter.py: This script is used to automatically split our dataset 'ProjectDatasets' <i>(different from the dataset in Part 1 of the project)</i> into 'train', 'validation', and 'test' folders. The images of our dataset are randomly divided into the folders with a 70%/15%/15% split respectively

		Step 1) Execute >python datasetsplitter.py  

<br />
c) <br />      
	-modeleval.py: This script is used to test our Main CNN model's capabilities of classifying images into 'Angry', 'Focused', 'Happy', or 'Neutral'. This script allows you to test an entire class' TEST dataset, an individual image from a class' TEST dataset, or a custom image the user can upload to the command line when running the program.

		Step 1) Execute >python modeleval.py  
		Step 2) Enter 1 to classify an individual image from the dataset OR enter 2 to classify a complete dataset OR enter 3 to classify a custom image 
		
		-IF OPTION 1 IS CHOSEN:
		Step 3) Enter the class name you want to classify (ex: "Angry")
		Step 4) Enter the image file name (including the file type) that you want to classify (must be an image taken from ./GeneratedSplitDataset/test/CLASS-PICKED-FROM-STEP3 ex: "1584.jpg" <- example is from ./GeneratedSplitDataset/test/Angry)
		
		-IF OPTION 2 IS CHOSEN:
		Step 3) Enter the class name you want to classify (ex: "Angry")
		
		-IF OPTION 3 IS CHOSEN:
		**NOTE:Only .png and .jpg type images are acceptable**
		Step 3) Enter the absolute path of the custom image you want to classify (ex: "C:\Users\Tristan\Pictures\customimage.jpg")
  
<br />
	-confusion_matrix_analysis_and_metrics.py: This script is used to calculate the Macro-Precision, Macro-Recall, Macro-F1, Micro-Precision, Micro-Recall, Micro-F1, and Accuracy values of the main model and two variants provided to the script, which then creates a corresponding confusion matrix

		Step 1) Execute >python confusion_matrix_analysis_and_metrics.py  
		Step 2) Enter the file name of the saved main model (ex: "main_best_model.pth")
		Step 3) Enter the file name of the saved variant 1 model (ex: "variant1.pth")
		Step 4) Enter the file name of the saved variant 2 model (ex: "variant2.pth")
		Note: You may enter the file names with or without the file type '.pth'
  
<br />
d) <br />      
	-cnnmodel_automated.py: CNNModel building script that builds the model based on various hyperparameters provided when calling the script's main method. Saves the created model configurations and metrics in a text file found in the ./models folder.
  <br />  
  <b>NOTE: This file is not meant to be run. Its script is called through the cnnmodel_single_gen.py and cnnmodel_permutation_gen.py</b>
<br />  
<br />
	-cnnmodel_single_gen.py: Script that takes user-inputted hyperparameters to use when calling the cnn_automated.py script's main method to create 1 model which is run 5 times 		
 		
		Step 1) Execute >python cnn_single_gen.py  
		Step 2) Enter the type of model to save (ex: 0 for main model, 1 for a variant)
		Step 3) Enter the hyperparameter values you would like to use with the given prompts
  
<br />
	-cnnmodel_permutation_gen.py: Automates the creation and saving of model configurations based on a large number of combination of hyperparameter values. Relies on the cnn_automated.py script to build the right models using the specified hyperparameters
 		
		Step 1) Execute >python cnn_permutation_gen.py  
 
## Dataset.pdf:  
<br />
	This file is unchanged from Part 1 of the project.

## ProjectReport-Part2.pdf:  
<br />
	This file contains the complete report of our work from Part 1 and Part 2 of our project.
<br /> <br />
 	The report includes a title page, information about the creation of our dataset, the explanation of our data cleaning techniques and challenges, the explanation of the labeling methods and 
	challenges, and the visualizations of our dataset. These visualizations include a bar graph to display the number of images our dataset 
	contains in each expression's training data class, one aggregrated pixel intensity distribution histogram for each expression's training 
	data class, and pixel intensity histograms for 15 randomly chosen images in each expression's training data class. Lastly, this report 
	contains a reference section for all the material used/referenced throughout this deliverable.  
<br /> <br /> <b>New additions from Part 2:</b> The report includes the model overview and details of our CNN architecture, it's training process, and the evaluations of the Performance metrics between the main model and the 2 variants, the confusion matrix analyses of 
  the 3 models, the impact of architectural variations, and our final conclusions and forward look of improvements for our model.

## Models Folder: 
<br />
	This folder contains the created model configurations and metrics text files generated from the cnnmodel_automated.py script. The complete contents of the models folder can be found in the 'phase2' branch of our repository

## model_count_config.txt: 
<br />
	This file contains the next model number that should appear in the next generated model text file inside the models folder. This text file is necessary for the functionality of the cnn_permutation_gen.py, cnn_single_gen.py scripts, and cnn_automated.py scripts

## Expectation of Originality files:  
<br />
	Expectations-of-Originality-June14-2024-AlinCaia.pdf: <br />
		This is the Expectation of Originality form signed by Alin Caia <br />
<br />
	Expectations-of-Originality-MelissaAnanian40112159.pdf: <br />
		This is the Expectation of Originality form signed by Melissa Ananian <br />
<br />
	Expectations-of-Originality-TristanManniMirolla40112168.pdf: <br />
		This is the Expectation of Originality form signed by Tristan Manni-Mirolla <br />
<br />
	These forms are provided to attest to the originality of our work.  
