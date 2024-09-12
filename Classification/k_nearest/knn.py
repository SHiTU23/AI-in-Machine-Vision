######################################
##  Classify images using KNN model
##  \\ inputs
##  \\ output
######################################


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from skimage import io, transform


def load_images(folder_path, image_size=(32, 32)):
    '''
    :param folder_path has sub-folders with names of labels
    '''
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)

        for file_name in os.listdir(label_folder):
            image_path = os.path.join(label_folder, file_name)
            image = io.imread(image_path)

            resized_image = transform.resize(image, image_size)
            images.append(resized_image.flatten()) ### make a 1D array of image
            labels.append(label)

    return np.array(images), np.array(labels)

images_path = 'images'
x, y = load_images(images_path) ## x: images y: labels
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.2, random_state=42)

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)

### the best prameters will be found in GridSearchCV
param_grid = {
                'n_neighbors': list(range(1, 50)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
             }
knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_gscv.fit(X_train, Y_train)
print(f"best K: {knn_gscv.best_params_}")

# y_pre = knn.predict(X_test)
y_pre = knn_gscv.predict(X_test)
accuracy = accuracy_score(Y_test, y_pre)
print(f"Accuracy of the model: {accuracy}")

new_image = 'cat.jpg'
image = io.imread(new_image)
image_resize = transform.resize(image, (32, 32))
image_flat = np.array(image_resize.flatten())
image_flat = image_flat.reshape(1, -1) ## reshape to (1, n_features)
# predict = knn.predict(image_flat)
predict = knn_gscv.predict(image_flat)
print(predict)
