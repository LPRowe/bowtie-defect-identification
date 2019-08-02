All files are named according to the order of the data in each row.

std0: standard deviation of shear 0 image
std45: standard deviation of shear 45 image
sh0-arr: shear 0 image (in list form)
sh45-arr: shear 45 image (in list form)
bow-bool: 1 if rows information is for a bowtie and 0 if for a nonbowtie
thetaM: angle of the circle sweep in the shear max image at which maximum intensity was measured
theta0: angle of the circle sweep in the shear 0 image at which maximum intensity was measured
theta45: angle of the circle sweep in the shear 45 image at which maximum intensity was measured
train: denotes npy file is intended to be used to train a ML classifier
144_dim: number of features included in the shear0 and shear 45 circle sweeps combined
	(in other words, twice the number of angles at which data was measured around each bowtie)
wafer: The wafer the bowtie is located on
loc: the image of the wafer the bowtie is located on
subloc: the index of the sub-image (after sub-dividing an image into 16 smaller arrays) the bowtie is located in
pixel: the index in the full image that is the center of the bowtie 
	image[pixel//x_dimension][pixel%x_dimension] <-- center of bowtie where x_dimension=640 (the width of an image)


For max-pooling SVC:
    std0_std45_sh0-arr_sh45-arr_bow-bool_train.npy
    wafer_loc_subloc_pixel_std0_std45_shear0_shear45_bow-bool.npy

For circle-sweep SVC:
    thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_144_dim.npy
    wafer_loc_subloc_pixel_thetaM_theta0_theta45_std0_std45_shear0_shear45_bow-bool.npy