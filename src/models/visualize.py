import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.callbacks import History
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import img_to_array, load_img
from keras import Model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import random


def plot_metrics(history: str, model: str) -> None:
    """
    Make loss curve and accuracy plot between each epoch and save in results/figures folder
    Arguments:
        history: History. History CSV name file containing loss and accuracy at each epoch outputted  from model.fit
        model: model name
    Returns:
        None
    """

    output_path = os.getcwd() + "/results/figures/"
    path = os.getcwd() + "/results/" + history

    hist = pd.read_csv(path)

    # summarize history for accuracy
    plt.figure()
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig(os.path.join(output_path, "acc_epochs_%s.jpg" % model))

    # summarize history for loss
    plt.figure()
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    plt.savefig(os.path.join(output_path, "loss_epochs_%s.jpg" % model))


def display_confusion_matrix(
    x_test: np.ndarray, y_test: np.ndarray, classes: dict, model_name: str
) -> None:
    """
    Make confusion matrix and normalized condusion matrix and save in results/figures folder
    Arguments:
        x_test: images form testing set (None, 32,32,3)
        y_test: labels from each image data in one-hot format
        classes: dictionary for each label
        model_name: model name svaed ing /models folder
    Returns:
        None
    """
    path = os.getcwd() + "/models"
    saved_path = os.getcwd() + "/results/figures"
    loaded_model = load_model(path + "/" + model_name)

    # calculate predicted values from model_name
    actual = np.argmax(y_test, axis=1)
    predicted = np.argmax(loaded_model.predict(x_test), axis=1)

    # compute confusion matrix
    cm = confusion_matrix(actual, predicted)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classes.values()
    )

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        cbar=True,
        xticklabels=classes.values(),
        yticklabels=classes.values(),
    )
    plt.xlabel("Predicted", size=10)
    plt.ylabel("Actual", size=10)
    plt.title("Confusion Matrix ", size=15)
    plt.show()
    plt.savefig(os.path.join(saved_path, "Confusion_matrix_%s_.jpg" % model_name))
    plt.show()
    # plt.close()


def plotmodel(model_name: str) -> None:
    path = os.getcwd() + "/results/figures/%s.jpg" % model_name
    model = load_model(os.path.join(os.getcwd() + "/models", model_name))
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=path)


def visualize_predictions(
    y_test: np.ndarray, x_test: np.ndarray, classes: dict, model_name: str
) -> None:
    """
    Visualize random images from traiing set and display score and true and predidcted class. save in results/figures folder
    Arguments:
        x_test: images form testing set (None, 32,32,3)
        y_test: labels from each image data in one-hot format
        classes: dictionary for each label
        model_name: model name svaed ing /models folder
    Returns:
        None
    """
    output_path = os.getcwd() + "/results/figures/"
    model = load_model(os.path.join(os.getcwd() + "/models", model_name))

    y_pred_test = model.predict(x_test)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)
    y_pred_test_max_probas = np.max(y_pred_test, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cols = 8
    rows = 4
    fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            random_index = np.random.randint(0, len(y_test))
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid("off")
            ax.axis("off")
            ax.imshow(x_test[random_index, :])
            pred_label = classes[y_pred_test_classes[random_index]]
            pred_proba = y_pred_test_max_probas[random_index]
            true_label = classes[y_test[random_index]]
            ax.set_title(
                "pred: {}\nscore: {:.3}\ntrue: {}".format(
                    pred_label, pred_proba, true_label
                )
            )
    plt.savefig(
        os.path.join(output_path, "Prediction visualizations %s.jpg" % model_name)
    )
    plt.show()


def unwrap_model(model):
    mobilenet = model.get_layer("mobilenet_1.00_224")
    inp = mobilenet.input
    out = mobilenet.output
    # out = model.get_layer('dropout')(mobilenet.output)
    # out = model.get_layer('dense')(out)
    return Model(inp, out)


def visualize(img: np.ndarray, rows: int, cols: int, model_name: str) -> None:
    """
    Visualize each feature map for a given image outputted from each convolutional layer and save in results/figures folder
    Arguments:
        img: input RGB image (32, 32, 3)
        rows : number of rows
        cols: nmber of columns
        model_name: model name svaed ing /models folder
    Returns:
        None
    """
    model = load_model(os.path.join(os.getcwd() + "/models", model_name))
    output_path = os.getcwd() + "/results/figures/"

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    # successive_outputs = [layer.output for layer in model.layers[2:]]
    last_conv_layer = "conv_pw_13_relu"
    unwrapped_model = unwrap_model(model)
    visualization_model = Model(
        inputs=unwrapped_model.inputs,
        outputs=[
            unwrapped_model.get_layer(last_conv_layer).output,
            unwrapped_model.output,
        ],
    )
    print(visualization_model.summary())
    # x = img_to_array(img)
    x = img.reshape((1,) + img.shape)
    resize_and_rescale = tf.keras.Sequential(
        [keras.layers.Resizing(224, 224), keras.layers.Rescaling(1.0 / 255)]
    )

    x = resize_and_rescale(x)

    plt.imshow(x[0, :, :, 0])
    plt.title("Input image")
    # plt.show()

    # Run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # Display representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map

            rows = np.ceil(n_features / cols).astype(int)
            fig = plt.figure(figsize=(cols, rows))
            j = 0
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x = np.clip(x, 0, 255).astype("uint8")
                # We'll tile each filter into this big horizontal grid

                ax1 = fig.add_subplot(rows, cols, i + 1)
                ax1.imshow(x, cmap="viridis")
                ax1.axis("off")
                ax1.set_title(str(i), fontsize=10)

            plt.suptitle("%s_%s" % (model_name, layer_name), fontsize=14)
            plt.tight_layout(pad=0.15, w_pad=1.0, h_pad=1.0)
            plt.savefig(
                os.path.join(
                    output_path,
                    "Output Feature Map %s %s.jpg" % (layer_name, model_name),
                )
            )
            plt.show()


if __name__ == "__main__":
    # visualize_layers()
    pass
