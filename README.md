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
* regarding the use of bowtie defects to characterize monocrystalline silicon wafers according to strength see [here](/documents/NDPE CHAR OF cSi PV WAFERS.pdf).
* about the methods used to collect and post process microscopic wafer images see [here](/documents/Overview of IR-GFP Wafer Image Data Set.pdf.ink)
* about the automated process of collecting microscopic shear stress measurements see [here](/documents/IR-GFP-automation.pdf).
* about the equipment used (IR-GFP) to image wafers see [here](/documents/IR-GFP.pdf)

Special thanks to Prof. Harley Johnson, Dr. Gavin Horn, Dr. Tung-wei Lin and Alex Kaczkowski whose guidance and advice have been deeply appreciated.

