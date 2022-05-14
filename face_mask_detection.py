from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import seaborn as sn
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


INIT_LR = 1e-4
EPOCHS = 10
BS = 32

imagePaths = list(paths.list_images("Train"))
print(imagePaths)
data = []
labels = []


def data_and_labels(imagePaths):
	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2]
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)
		data.append(image)
		labels.append(label)

	return data, labels


def one_hot_encoding(labels):
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)	
	labels = to_categorical(labels)

	return labels, lb.classes_


def data_splitting(data, labels):
	trainX, testX, trainY, testY = train_test_split(data, labels,
		test_size=0.20, stratify=labels, random_state=42)

	return trainX, testX, trainY, testY


def data_augmentation():
	aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

	return aug


def mobile_net_v2():
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))

	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	for layer in baseModel.layers:
		layer.trainable = False

	model = Model(inputs=baseModel.input, outputs=headModel)

	return model


def model_compilation(model):
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	return model


def model_training(model):
	print("[INFO] training head...")
	history = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch=len(trainX) // BS,
		validation_data=(testX, testY),
		validation_steps=len(testX) // BS,
		epochs=EPOCHS)

	return model, history


def model_testing(model, testX):
	predictions = model.predict(testX, batch_size=BS)
	predictions = np.argmax(predictions, axis=1)

	return predictions


def plot_confusion_matrix(testY, predictions):
	cf_matrix = confusion_matrix(testY.argmax(axis=1), predictions)

	df_cm = pd.DataFrame(cf_matrix, index = [i for i in ["with_mask", "without_mask"]],
                  columns = [i for i in ["with_mask", "without_mask"]])

	plt.figure(figsize = (8,8))
	sn.heatmap(df_cm, annot=True)
	plt.show()


def model_classification_report(testY, predictions, classes):
	print(classification_report(testY.argmax(axis=1), predictions,
	target_names=classes))


def plot_graph(H):
	N = EPOCHS
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("graph")





data, labels = data_and_labels(imagePaths)

data = np.array(data, dtype="float32")
labels = np.array(labels)

labels, classes = one_hot_encoding(labels)

trainX, testX, trainY, testY = data_splitting(data, labels)

aug = data_augmentation()

model = mobile_net_v2()

model = model_compilation(model)

model, history = model_training(model)

predictions = model_testing(model, testX)

plot_confusion_matrix(testY, predictions)

model_classification_report(testY, predictions, classes)

plot_graph(history)



