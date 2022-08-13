import pandas
import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from os.path import join
import random
import pickle
#==============
print("data_prep file executed")
#==============

data_folder = "data"
print(f"the current folder is : {os.getcwd()}")


def prepare_all_images():
    files_list = os.listdir(data_folder)
    resolu = 3
    WIDTH = 32*resolu
    LENGTH = 32*resolu
    dim = (WIDTH,LENGTH)
    all_images = []
    for filee in files_list :
        if 'png' in filee :
            im_arry = cv2.imread(join(data_folder,filee))
            im_arry = cv2.resize(im_arry,dim,cv2.INTER_LINEAR)
            all_images.append(np.asarray(im_arry))
    all_images = np.reshape(all_images,(-1,WIDTH,
            LENGTH,3))
    all_images = all_images.astype(np.float32)
    all_images = all_images / 127.5 - 1.
    with open(join(data_folder,"training_data.pkl") , 'wb') as f :
        pickle.dump(all_images,f)


def get_rand_pok():
    files_list = os.listdir(data_folder)
    filee = random.choice(files_list)
    im_arry = cv2.imread(join(data_folder,filee))
    print(f"resolution of the image is : {im_arry.shape}")
    return im_arry

def load_data() :
    with open(join(data_folder,"training_data.pkl") , 'rb') as f :
        data = pickle.load(f)
    return data

if __name__ == '__main__' :
    prepare_all_images()







