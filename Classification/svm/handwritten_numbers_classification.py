import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

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
display.display_images("training", rows=2, columns= 10,
                images_list=digits.images, labels_list=labels)

### flatten the images
n_sample = len(digits.images)
data = digits.images.reshape(n_sample, -1)

### Support Vector Classifier
classifier = svm.SVC(gamma=0.001)
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.5, shuffle=False)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)
display.display_images("prediction", rows=2, columns=10, 
                images_list=X_test, labels_list=prediction)

### report
print(f"classifier report for {classifier}\n"
      f"{metrics.classification_report(y_test, prediction)}")

### confusion matrix
confusionMatrix_disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, prediction)
confusionMatrix_disp.figure_.suptitle("SVM Confusion Matrix")
print(f"confusion Matrix:\n{confusionMatrix_disp.confusion_matrix}")

display.show()
