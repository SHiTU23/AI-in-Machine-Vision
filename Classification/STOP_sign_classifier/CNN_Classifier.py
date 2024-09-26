import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

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

##################################################
###                 TRAIN DATA                 ###
##################################################
model = Sequential()

#####################################
###           ADD LAYERS          ###
#####################################
### First layer:
"""
 adding convolutional layer and max pooling layer
 the first layer needs an input
 the first convolution has 16 filters of 3 pixels by 3pixels in size
 and these filters are going to move 1 pixel each time
 input_shape is the size of images with 3 color channel 
"""
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D()) ## take out the max value of the 2 by 2 region

### Second layer:
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

### Third layer:
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

### we want to reach to a single value. So dont want the convolutional channel values
model.add(Flatten())

### Fully connected layers
model.add(Dense(256, activation='relu'))
### Final layer: map the output into 0 and 1
model.add(Dense(1, activation='sigmoid'))

### adam is a tf.optimazer
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
print(model.summary())

#####################################
###           TRAIN DATA          ###
#####################################
logdir = os.path.join(currect_script_path, 'logs')
## save check points and logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
print(hist)

#####################################
###      PLOT THE PERFORMANCE     ###
#####################################
"""
loss_fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
loss_fig.suptitle("Loss", fontsize=20)
plt.legend(loc="upper left")
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle("Accuracy", fontsize=20)
plt.legend(loc="upper left")
plt.show()
"""

##################################################
###             Evaluate the model             ###
##################################################
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)
### by having 1 for all of them, the model is working good
print(f"Precision: {precision.result().numpy()}, Recall: {recall.result().numpy()}, Accuracy: {accuracy.result().numpy()}")