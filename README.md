------------------------------------------------------------------------------------------------------
Group name: FS-4
Group members: Tristan Manni-Mirolla (40112168), Alin Caia (40246955), Melissa Ananian (40112159)
------------------------------------------------------------------------------------------------------

The following packages must be installed in your environment to execute the Python scripts that we wrote for this project:   

	>pip install pillow  
	>conda install scikit-learn  
	>pip install matplotlib OR conda install matplotlib  
	>pip install opencv-python==4.5.3.56  

For OpenCV, using the latest version was not compatible with Python 3.6, so we installed a previous version that was found to work online.  
All the Python scripts prompt the user to enter an absolute file path to a dataset in the format 'C:\Users\Tristan\Documents\Concordia\COMP472\Project-Part1'.  

To execute our Python scripts on our dataset, you must first download our dataset from our GitHub Repository.  
Direct link: https://github.com/serbancaia/ai-comp472/blob/ce8d08fd58db7d4b2cab80755f3ebc5cf968d160/ProjectDatasets.zip  

Our '.py' Python script files:  
---STEPS FOR EXECUTING OUR CODE FOR (a) DATA CLEANING AND (b) DATA VISUALIZATION---  
Step 0) Open the Python environment with the above packages installed through the command prompt and navigate to the directory containing the python scripts.  
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

Dataset.pdf: <br />
	This file contains the provenance of the datasets and images used for our project's dataset. Referencing of the online dataset and  
	their licensing type is listed. This document also showcases the custom images our team created for each class for our dataset along  
	with the necessary modifications for standardization with the rest of our dataset. Lastly, this file contains a sample set of 25  
	representative images for each class in our dataset.  
	A direct link to our dataset's .zip file (uploaded to our GitHub repository) is provided at the top of this file.  
	The purpose of this file is to display information about our dataset and the online datasets we used as well as showcase some of the  
	images that could be found in our dataset, as we are unable to upload the entire dataset to Moodle due to its large size.  
<br />
ProjectReport-Part1.pdf:  
	This file contains the complete report of our work for Part 1 of our project. The report includes a title page, information about the   
	creation of our dataset, the explanation of our data cleaning techniques and challenges, the explanation of the labeling methods and   
	challenges, and the visualizations of our dataset. These visualizations include a bar graph to display the number of images our dataset  
	contains in each expression's training data class, one aggregrated pixel intensity distribution histogram for each expression's training   
	data class, and pixel intensity histograms for 15 randomly chosen images in each expression's training data class. Lastly, this report   
	contains a reference section for all the material used/referenced throughout this deliverable.  
<br />
Expectation of Originality files:  
	Expectations-of-Originality-May30-2024-AlinCaia.pdf:  
		This is the Expectation of Originality form signed by Alin Caia  
	Expectations-of-Originality-MelissaAnanian40112159.pdf:  
		This is the Expectation of Originality form signed by Melissa Ananian  
	Expectations-of-Originality-TristanManniMirolla40112168.pdf:  
		This is the Expectation of Originality form signed by Tristan Manni-Mirolla  
  <br />
	These forms are provided to attest to the originality of our work.  
