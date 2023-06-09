import cv2             # OpenCV computer vision for image processing
import os              # interacting with operating system
import json            # Javascript Mainly for config files, storing data or APIs (for working with JSON files)
from tqdm import tqdm  # For displaying progress bars during iterations
import numpy as np     # For numerical operations
import sys


# MAIN CODE STARTS HERE
main_dir = '/Users/seanl/OneDrive/Desktop/Pytorch_Project' # Linux is forward slash /
#save_gt_folder = '/Users/seanl/OneDrive/Desktop/Pytorch_Project/Groundtruth'
name_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'J':18,'K':19,'L':20,'M':21,'N':22,'P':23,'Q':24,'R':25,'S':26,'T':27,'U':28,'V':29,'W':30,'X':31,'Y':32,'Z':33}
dataset = 'HTX CARS DATASET 10'
labelset = "Labels Set 10.json"
# with open(os.path.join(main_dir,"Labels Set 1.json"),'r') as file:
#     data = json.load(file)

#print(data)
#filename should be in string
for filename in os.listdir(os.path.join(main_dir, dataset)):
    #print(filename)
    with open(os.path.join(main_dir, labelset),'r') as file:
        data = json.load(file)

    name = data[filename]['filename']
    #print(name)
    all_x_coord=(data[filename]['regions']['0']['shape_attributes']['all_points_x'])
    #print(all_x_coordinates)
    all_y_coord=(data[filename]['regions']['0']['shape_attributes']['all_points_y'])
    #print(all_y_coordinates)

    complete_coordinates = f"{int(all_x_coord[0])}&{int(all_y_coord[0])}_{int(all_x_coord[1])}&{int(all_y_coord[1])}_{int(all_x_coord[2])}&{int(all_y_coord[2])}_{int(all_x_coord[3])}&{int(all_y_coord[3])}"
    #int() to remove the decimals
    #print(complete_coordinates)

    lp_format = ''
    for char in filename[:-4]: # -4 to remove the .jpg
        #print(char)
        lp_format=lp_format + f"{name_dict[char]}_"
    lp_format=lp_format[:-1] #remove the last _
    #print(lp_format)

    complete_filename = f"{complete_coordinates}-{lp_format}-A.jpg"
    #print(complete_filename)

    current_file_path = os.path.join(main_dir,dataset,filename)
    #print(current_file_path)
    new_file_path = os.path.join(main_dir,dataset,complete_filename)
    #print(new_file_path)

    os.rename(current_file_path,new_file_path)
