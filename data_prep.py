import pandas
import cv2
import numpy
import os
from glob import glob
import matplotlib.pyplot as plt
from os.path import join
import random
#==============
print("data_prep file executed")
#==============

data_folder = "data"
print(f"the current folder is : {os.getcwd()}")

files_list = os.listdir(data_folder)
print(f"the first file is : \n {files_list[0]}")

def get_rand_pok():
    files_list = os.listdir(data_folder)
    filee = random.choice(files_list)
    im_arry = cv2.imread(join(data_folder,filee))
    return im_arry


