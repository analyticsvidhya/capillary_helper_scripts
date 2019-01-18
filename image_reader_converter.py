### This script creates the dataset from the images which is suitable to feed to the autoencoder input. It also resizes the data to 128x128

import numpy as np
import csv
import os
import math
images_folder_path = '../train/images'
import cv2
import IPython

def resize_image(img, dim):
    padding = (img.shape[0] - img.shape[1]) / 2
    img=cv2.copyMakeBorder(img, top=0, bottom=0, left=math.floor(padding), right=math.ceil(padding), borderType= cv2.BORDER_CONSTANT, value=[dim,dim,dim])
    # print(img.shape)    
    resized_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
    return resized_img

image_list = os.listdir(images_folder_path)
final_images = list()
image_id_list = list()
for image_name in image_list:
    image_id = image_name.split(".")[0]
    image_id_list.append(int(image_id))
    image = cv2.imread(os.path.join(images_folder_path, image_name))
    resized_img = resize_image(image, 128)
    final_images.append(resized_img)
images_dataset = np.stack(final_images)
np.save('images_dataset.npy', images_dataset)

product_id_file = "product_image_id_list.csv"

with open(product_id_file, 'w') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter='\n')
     wr.writerow(image_id_list)
