# version (0 for SAM-VGG and 1 for SAM-ResNet)
version = 1
# batch size
b_s = 8
# number of rows of input images
shape_r = 240
# number of cols of input images
shape_c = 320
# number of rows of downsampled maps

shape_r_gt = 30
# number of cols of downsampled maps
shape_c_gt = 40

# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640

# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 10
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16