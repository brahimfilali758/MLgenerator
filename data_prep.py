import pandas
import cv2
import numpy
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
    resolu = 2
    WIDTH = 32*resolu
    LENGTH = 32*resolu
    dim = (WIDTH,LENGTH)
    all_images = []
    for filee in files_list :
        im_arry = cv2.imread(join(data_folder,filee))
        im_arry = cv2.resize(im_arry,dim,cv2.INTER_LINEAR)
        all_images.append(im_arry)
    with open(join(data_folder,"training_data.pkl") , 'wb') as f :
        pickle.dump(all_images,f)


def get_rand_pok():
    files_list = os.listdir(data_folder)
    filee = random.choice(files_list)
    im_arry = cv2.imread(join(data_folder,filee))
    print(f"resolution of the image is : {im_arry.shape}")
    return im_arry


if __name__ == '__main__' :
    pass







