# bowtie-defect-identification
Bowtie defects (shown below) can be observed in microscopic shear stress images of silicon wafers.  
They are the result of residual stress acting on microcracks.  

![Example Bowtie Defect](images/0009raw_1188.png)

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
The .py files provided in this repository are intended to be run sequentially according to the number in the file name.  The .dt1 files containing 3,234-3,500 microscopic IR-GFP images for each wafer are not included in the repository.  You can access them through the lab external hard drive or copy them into the Wafer Data Files folder included in the repository.  If you choose to do the latter then update the variables datafile and dt1file in the code to reflect the new location of the .dt1 files.  

1. Collects light level information from each microscopic image and tracks which pixels have uncommonly high intensity.  This information is saved in the designated savefile and will be used to filter out low quality images and identify hypersensitive pixels.  
    1. Set post-saw-damage-removal boolean.
    * True if wafer name starts with 00 or 02 and False otherwise
    1. Update savefile and datafile if necessary
