import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

class display:
    def display_images(title, rows, columns, images_list, labels_list):

        ### show the first 5 digits
        _, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 3))
        image_index=0
        if rows > 1:
            for row in axes:
                for ax, image, label in zip(row, images_list[image_index:], labels_list[image_index:]):
                    image = image.reshape(8, 8)
                    ax.set_axis_off()
                    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
                    ax.set_title(f"{title} {label}")
                    image_index += 1
        else:
            for ax, image, label in zip(axes, images_list[image_index:], labels_list[image_index:]):
                    image = image.reshape(8, 8)
                    ax.set_axis_off()
                    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
                    ax.set_title(f"{title} {label}")
                    image_index += 1
    def show():
        plt.tight_layout()  ### avoid overlaping
        plt.show()

### raw data
digits = datasets.load_digits()
labels = digits.target

### flatten the images
n_sample = len(digits.images)
data = digits.images.reshape(n_sample, -1)
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.5, shuffle=False)

#######################################
###               SVM               ###
#######################################
svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(X_train, y_train)
svm_prediction = svm_classifier.predict(X_test)

### SVM report
print(f"svm_classifier report for {svm_classifier}\n"
      f"{metrics.classification_report(y_test, svm_prediction)}")

### SVM confusion matrix
svm_confusionMatrix = metrics.ConfusionMatrixDisplay.from_predictions(y_test, svm_prediction)
svm_confusionMatrix.figure_.suptitle("SVM Confusion Matrix")

#######################################
###               KNN               ###
#######################################
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_prediction = knn_classifier.predict(X_test)

### KNN report
print(f"knn_classifier report for {knn_classifier}\n"
      f"{metrics.classification_report(y_test, knn_prediction)}")

### KNN Confusion Matrix
knn_confusionMatrix = metrics.ConfusionMatrixDisplay.from_predictions(y_test, knn_prediction)
knn_confusionMatrix.figure_.suptitle("KNN Confusion Matrix")

display.show()
