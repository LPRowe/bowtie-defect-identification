# bowtie-defect-identification
Bowtie defects (shown below) can be observed in microscopic shear stress images of silicon wafers.  
They are the result of residual stress acting on microcracks.  

<p align='center'><img src='images/0009raw_1188.png' width='600'></p>

This repository will address:
1. Converting .dt1 (Delta Vision) files into Shear 0, Shear 45, and Shear Max images
1. Creating/applying subtraction images
1. Detecting (and removing) hypersensitive pixels and unnecessary images
1. Cleaning manually identified bowties and non-bowties
1. Building a machine learning classifier using SVM and feature engineering

For information:
* regarding the use of bowtie defects to characterize monocrystalline silicon wafers according to strength see [here](/documents/NDPE%20CHAR%20OF%20cSi%20PV%20WAFERS.pdf).
* about the methods used to collect and post process microscopic wafer images see [here](/documents/Overview_IR-GFP_Wafer_Image_Data_Set.pdf)
* about the automated process of collecting microscopic shear stress measurements see [here](/documents/IR-GFP-automation.pdf).
* about the equipment used (IR-GFP) to image wafers see [Horn et al. "Infrared grey-field polariscope: A tool for rapid stress analysis
in microelectronic materials and devices" Review of Scientific Instruments 2005.](/documents/Horn_2005.pdf) and J. Lesniak & M. Zickel, "Applications of Automated Grey-Field Polariscope" SEM June 1998.

Special thanks to Prof. Harley Johnson, Dr. Gavin Horn, Dr. Tung-wei Lin and Alex Kaczkowski whose guidance and advice have been deeply appreciated.

# Instructions
The .py files provided in this repository are intended to be run sequentially according to the number in the file name; points 1 - 6 below correspond to .py files beginning with 1 - 6.  The .dt1 files containing 3,234-3,500 microscopic IR-GFP images for each wafer are not included in the repository.  You can access them through the lab external hard drive or copy them into the Wafer Data Files folder included in the repository.  If you choose to do the latter then update the variables datafile and dt1file in the code to reflect the new location of the .dt1 files.  

1. Collects light level information from each microscopic image and tracks which pixels have uncommonly high intensity.  This information is saved in the designated savefile and will be used to filter out low quality images and identify hypersensitive pixels.  
    1. Set post-saw-damage-removal boolean.
        * True if wafer name starts with 00 or 02 and False otherwise
    1. Update savefile and datafile if necessary
    1. If running multiple wafers at once, uncomment multi-wafer analysis section and comment out waferlist
1. Records the average (across a full wafer) value and standard deviation (across a full wafer) of the average light (within each microscopic image) and standard deviation (within each microscopic image) of pixel intensities.  These values will be used to filter out low quality microscopic IR-GFP images based on light level and standard deviation of pixel intensities.  
    1. Update datafile and save file
1. During long imaging sessions, the detector heats up.  When the detector is hot it often has hypersensitive (hot) pixels.  These are pixels which register higher than true shear max values, which likely means their light intensity flickers.  This script identifies and records the location of hypersensitive pixels.  More details are included within the script.  
    1. Update file locations
    1. Set psdr boolean
    1. Update waferlist to reflect which wafer(s) you want to check for hot pixels
    * A hypersensitive pixel is one that occurs as the most intense pixel (in a shear max image) 4 or more times out of 3,234 shear max iamges on a wafer.  Each image contains 640 * 480 pixels, as such the probability of a single pixel occuring that has already occured as the most intense pixel in an image reoccuring as the most intense pixel in 1 to 5 other images is given by the following graph. 
    '''
    <p align='center'><img src='images/hypersensitive-pixel-probability.JPG' width='400'></p>
    '''
1. Generates a shear 0 and shear 45 subtraction image for each wafer.  A subtraction image is the average of 108 images and will be used to remove the effects of consistent optical aberrations caused by reflections, imperfections in the optic train, and a polychromatic light source.  Low quality images (images of the wafer mask or blurry images) are filtered out prior to selecting the 108 images that will be used to make the subtraction image.  Of the remaining images, the 108 images with the lowest average shear max retardation are selected to make the subtraction image.  A brief explanation for why this is done is included in the script.   
    <p align='center'><b>Example Subtraction Image</b></p>
    <p align='center'><img src='images/shear_0_45_subtraction_image.png' width='600'></p>
    <p align='center'><b>Effect of Applying Subtraction Image</b></p>
    <p align='center'><img src='images/0009raw_93_SubExample.png' width='600'></p>
1. Generates annotated images of 5x IR-GFP images to facilitate the process of manually identifying bowties to train the machine learning classifier.  The procedure of doing so is as follows:
    1. Each image is generated from a .dt1 file
    1. Low quality images (blurry or low light) are filtered out
    1. Images are post processed by applying a subtraction image and removing hypersensitive pixels
    1. Shear max images are sub-divided into 16 equally sized "sub-images"
    1. The most intense pixel in the shear max sub-image region is boxed and annotated by the sub-image index
    1. Image is saved for manual bowtie identification
    <p align='center'><b>Shear 0 Manual Bowtie Identification</b></p>
    <p align='center'><img src='images/285_0.png' width='600' title='Shear 0 Manual Bowtie Identification'></p>
    <p align='center'><b>Shear 45 Manual Bowtie Identification</b></p>
    <p align='center'><img src='images/285_45.png' width='600' title='Shear 45 Manual Bowtie Identification'></p>
    In the example there is a large bowtie at location 3 and smaller bowties at locations 7 and 12. For your convenience set of approximately 500 manually identified bowties and non-bowties are already provided here ([non-bowties](https://github.com/LPRowe/bowtie-defect-identification/tree/master/Wafer_Images/manually_identified_non-bowties) & [bowties](https://github.com/LPRowe/bowtie-defect-identification/tree/master/Wafer_Images/manually_identified_bowties))
    
    
