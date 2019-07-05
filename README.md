# bowtie-defect-identification
Bowtie defects (shown below) can be observed in microscopic shear stress images of silicon wafers.  
They are the result of residual stress acting on microcracks.  

![Example Bowtie Defect](/images/example_bowtie.png)

This repository will address:
1. Converting .dv1 (Delta Vision) images into Shear 0, Shear 45, and Shear Max images
1. Creating/applying subtraction images
1. Detecting (and removing) hypersensitive pixels and unnecessary images
1. Cleaning manually identified bowties and non-bowties
1. Building a machine learning classifier using SVM and feature engineering

For information:
* regarding the use of bowtie defects to characterize monocrystalline silicon wafers according to strength see [here](/documents/ndpe_characterization_si_pv_wafers.pdf).
* about the automated process of collecting microscopic shear stress measurements see [here](/documents/IR-GFP-automation.pdf).
* about the equipment used (IR-GFP) to image wafers see [here](/documents/IR-GFP.pdf)

Special thanks go out to Prof. Harley Johnson, Dr. Gavin Horn, Dr. Tung-wei Lin and Alex Kaczkowski whose guidance and advice have been deeply appreciated.

