# -*- coding: utf-8 -*-

# this snippt of code is used to predict image categories

from keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import os


def predict(new_images_dir,img_rows,img_cols):
    
    files = os.listdir(new_images_dir)
    for file in files:    
        dest_file = new_images_dir + '/' + file
        newImage =cv2.imread(dest_file)    
        img = Image.fromarray(newImage)
        im = img.resize((img_rows,img_cols))
        imtoarr = img_to_array(im)
        imtoarr = imtoarr.reshape(1,3,img_rows,img_cols)
        class_index = model.predict_classes(imtoarr)
        new_val = class_index[0]    
        print("Given: %s Predicted: %s " %(file,new_val[0]))
