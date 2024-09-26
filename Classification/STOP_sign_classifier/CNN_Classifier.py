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
"""
print(batch[0].max()) ### max number of intestity which should be 1 after normalizing
print(len(batch)) ### 2 : images and lables 
"""

### show 4 of the images in the batch
"""
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
plt.show()
"""

##################################################
###                 SPLIT DATA                 ###
##################################################
number_of_batches = len(data)
print(number_of_batches) ### there is 7 batches of data.
### after spliting data, overall there should be 7 batches again. dont miss any

train_size = int(number_of_batches*.7) ### 70% of data
val_size = int(number_of_batches*.2)+1 ### 20% of data
test_size = int(number_of_batches*.1)+1 ### 10% of data
print(f"train size: {train_size}, val size: {val_size}, test size: {test_size}")
print(number_of_batches == (train_size + val_size + test_size)) ### SHOULD BE TRUE

train = data.take(train_size) ### allocate 4 batches of data to train
val = data.skip(train_size).take(val_size) ### skip the batches that was alocated to train
test = data.skip(train_size + val_size).take(test_size)
print(len(train))