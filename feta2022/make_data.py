import nilearn.image as ni
import glob
from tqdm import tqdm
from scipy.ndimage import zoom
import h5py
import numpy as np
import os

fetadata_dir = '/project/ajoshi_27/feta_2022/feta_2.2/'
mode = 'valid'
out_h5file = mode+'_t2.h5'

# Read the list of subjects
with open(mode+'.txt', 'r') as myfile:
    sub_lst = myfile.read().splitlines()

patch_size = [256, 256]
X = list()
Y = list()

for subname in tqdm(sub_lst):

    # Create file names
    mr = subname + '_T2w.nii.gz'
    lab = subname + '_dseg.nii.gz'

    mr_data = ni.load_img(mr).get_fdata()
    lab_data = ni.load_img(lab).get_fdata()

    for i in range(mr_data.shape[2]):
        slice = zoom(mr_data[:, :, i], (patch_size[0] /
                     mr_data.shape[0], patch_size[1]/mr_data.shape[1]), order=3)
        labels = zoom(lab_data[:, :, i], (patch_size[0] /
                      mr_data.shape[0], patch_size[1]/mr_data.shape[1]), order=0)
        slice[slice < 0] = 0

        if np.max(np.uint8(labels))>7 or np.min(np.uint8(labels))<0:
            print('Bad Label, skipping')
            continue

        X.append(np.uint8(255.0*slice/slice.max()))
        Y.append(np.uint8(labels))




hf = h5py.File(out_h5file, 'w')
hf.create_dataset('X', data=X)
hf.create_dataset('Y', data=Y)
hf.close()
