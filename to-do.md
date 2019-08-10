# <u>To Do</u>
* [x]  Edit scripts and supporting modules to run on Python 3.7
* [x]  Structure directory and rename file paths for Git repository
* [x]  Upload scripts to read .dt1 files
* [x]  Upload scripts to post process 5x images (subtraction, light filter, & hot pixels)
* [x]  Upload scripts to annotate potential bowties
* [x]  Manually identify (non)bowties from annotated images
* [x]  Upload script to crop and save .npy files for manually identified bowties and non-bowties
* [ ]  Upload script (and supporting modules) to clean .npy files and store in pandas DF
* [ ]  Extend master branch with proposed transformation (circle scan) 
* [ ]  Create a branch for each classifier (SVM (linear, polynomial, RBF), Random Forest, & KNeighbors)
* [ ]  Test precision and recall of each branch


* [x]  Create script to save images of non-bowties
* [ ]  Create script to crop all non-bowties and bowties, save as .npy in 'Wafer_Images\\Bowties\\Wafer' and 'Wafer_Images\\non-bowties\\wafer'
* [x]  Manually identify bowties for wafer 27


* [ ]  Save all circle scans of shear 0 and shear 45 to pd data file


* [ ]  Make tool to plot parametric plot given wafer, location to assist with identifying bowties
* [ ]  Add example bowtie and non-bowtie image, circle scan, and parametric plot pair to README.md
* [ ]  Add GIF of shear 0 shear 45 bowtie to show how to recognize a bowtie
* [ ]  Add a GIF of shear 0 and shear 45 dust to show how to recognize dust


* [ ]  Gaussian low pass filter and subtract from image (equivalent high pass filter)
* [ ]  Standard deviation as feature for bowtie training set

* [ ]  Blender Probabilities: https://stats.stackexchange.com/questions/155817/combining-probabilities-information-from-different-sources