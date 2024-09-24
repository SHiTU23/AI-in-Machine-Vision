import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt

#######################################
###             LOAD DATA           ###
#######################################
labels_list = []
images_list = []

current_script_path = os.path.dirname(__file__)
groups_dir = (os.path.join(current_script_path, 'images'))
print(groups_dir)
for label in os.listdir(groups_dir):  
    label_dir = os.path.join(groups_dir, label)
    print(label_dir)
    for file_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, file_name)
        ## open image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"image {file_name} read")

        resized_image = cv2.resize(image, (256, 256))
        print("image resized")
        labels_list.append(label)
        images_list.append(resized_image.flatten())

### convert to np array
labels = np.array(labels_list)
images = np.array(images_list)
print("DATA LOADED COMPLETELY")

#######################################
###         TRAIN TEST SPLIT        ###
#######################################
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=42)
print("DATA SPLITED IN TRAIN & TEST")

#######################################
###            TRAIN DATA           ###
#######################################
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
print("MODEL IS TRAINED")

#######################################
###              PREDICT            ###
#######################################
y_hat = knn_classifier.predict(X_test)
print("MODEL EVALUATED")

#######################################
###              REPORT             ###
#######################################
report = metrics.classification_report(y_test, y_hat)
print(f" REPORT FOR {knn_classifier}\n"
      f"{report}")

#######################################
###               TEST              ###
#######################################
test_image_path = os.path.join(current_script_path, 'stop.jpg')
test_im = cv2.resize(cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE), (256, 256))
test_image_flat = (test_im.flatten()).reshape(1, -1)
new_yhat = knn_classifier.predict(test_image_flat)

plt.figure()
plt.imshow(test_im, cmap='gray')
plt.title(f"prediction: {new_yhat}")
plt.show()