import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

currect_script_path = os.path.dirname(__file__)
images_dir_path = os.path.join(currect_script_path, 'images')

##################################################
###     Clear dataset from other extensions    ###
##################################################
image_exts = ['jpeg', 'jpg', 'png', 'bmp']
for label in os.listdir(images_dir_path):
    label_path = os.path.join(images_dir_path, label)
    for image_name in os.listdir(label_path):
        print(image_name)
        image_extention = (image_name.split(".")[1]).lower() ### take the second part
        
        ### if the extention is not in the list, remove th image
        if image_extention not in image_exts:
            image_path = os.path.join(label_path, image_name)
            os.remove(image_path)
            print(f"removed image {image_name}")

##################################################
###                  LOAD DATA                 ###
##################################################
### this will batch the dataset into 32, resize images to (256, 256), shuffle them
data = tf.keras.utils.image_dataset_from_directory(images_dir_path) 

#####################################
###          SCALING DATA         ###
#####################################
### scaling the intensity of pixels and normolize it 
### it means we map the colors from 0-255 to 0-1 but the image will still be coloful
scaled_data = data.map(lambda x, y: (x/255, y))

### we cannot iterate on data right now, because it has only created the data
data_iterator = scaled_data.as_numpy_iterator()

### grabing one batch of data
batch = data_iterator.next()
print(batch[0].max()) ### max number of intestity which should be 1 after normalizing
print(len(batch)) ### 2 : images and lables 

### show 4 of the images in the batch
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
plt.show()
