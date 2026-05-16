import os
import numpy as np

# dataset path
dataset_path = r"C:\Users\viraj\OneDrive\p60\Assignment\part2_cnn_project\images"

# show class names
classes = os.listdir(dataset_path)

print("Classes in dataset:")
print(classes)
from PIL import Image
import matplotlib.pyplot as plt

# choose one class
class_name = "dent"

# path to dent folder
class_path = os.path.join(dataset_path, class_name)

# first image
image_name = os.listdir(class_path)[0]

# full image path
image_path = os.path.join(class_path, image_name)

# open image
img = Image.open(image_path)

# show image
plt.imshow(img)
plt.title(class_name)
plt.axis("off")
plt.show()
print("\nNumber of images in each class:")

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    # skip non-folder files like .DS_Store
    if os.path.isdir(class_path):
        image_count = len(os.listdir(class_path))
        print(class_name, ":", image_count)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# image size
img_size = (128, 128)

# empty lists
X = []
y = []

# loop through classes
for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    # skip non-folder files
    if not os.path.isdir(class_path):
        continue

    # loop through images
    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        # load image
        img = load_img(image_path, target_size=img_size)

        # convert image to array
        img_array = img_to_array(img)

        # normalize pixels
        img_array = img_array / 255.0

        # store image
        X.append(img_array)

        # store label
        y.append(class_name)

print("Images loaded successfully")
print("Total images:", len(X))
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# convert X list to numpy array
X = np.array(X)

# convert text labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# convert numbers into categorical format
y_categorical = to_categorical(y_encoded)

print("X shape:", X.shape)
print("y shape:", y_categorical.shape)
print("Classes:", label_encoder.classes_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_categorical
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(4, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# predictions
y_pred = model.predict(X_test)

# convert probabilities to class numbers
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=label_encoder.classes_
))
# sample predictions
for i in range(5):
    plt.imshow(X_test[i])
    plt.title(
        "Actual: " + label_encoder.classes_[y_true_classes[i]] +
        " | Predicted: " + label_encoder.classes_[y_pred_classes[i]]
    )
    plt.axis("off")
    plt.show()