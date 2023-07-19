import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from BreastCancerClassifier.breast_cancer_classifier import CancerNet
from BreastCancerClassifier import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

NUM_EPOCHS = 30
BS = 32
INIT_LR = 1e-2

trainPaths = list(paths.list_images(config.TRAIN_PATH))

trainLabels = np.array([int(p.split(os.path.sep)[-2]) for p in trainPaths])
trainLabels = to_categorical(trainLabels)

classTotals = trainLabels.sum(axis=0)
print("classTotals:", classTotals)

classWeight = compute_class_weight('balanced', classes=np.unique(trainLabels.argmax(axis=1)), y=trainLabels.argmax(axis=1))

trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

valAug = ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS
)

valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS
)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS
)

model = CancerNet.build(width=48, height=48, depth=3, classes=2)
opt = tf.keras.optimizers.Adagrad(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Changed the class_weights dictionary to map the class labels to their corresponding weights.
class_weights = {0: 1, 1: 2}

lenTrain = len(trainPaths)
lenVal = len(list(paths.list_images(config.VAL_PATH)))
lenTest = len(list(paths.list_images(config.TEST_PATH)))

M = model.fit(
    trainGen,
    steps_per_epoch=lenTrain // BS,
    validation_data=valGen,
    validation_steps=lenVal // BS,
    class_weight=class_weights,
    epochs=NUM_EPOCHS
)

print("Now evaluating the model")
pred_probs = model.predict(testGen, steps=lenTest // BS + 1)
pred_indices = np.argmax(pred_probs, axis=1)

print(classification_report(testGen.classes, pred_indices, target_names=testGen.class_indices.keys()))

cm = confusion_matrix(testGen.classes, pred_indices)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')


