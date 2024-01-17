import random
import time
from torch.utils.data import Dataset
from torch_snippets import read, randint
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

random.seed(time.time())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
id2int = {'Parasitized': 0, 'Uninfected': 1}


class Malaria(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        for key in id2int.keys():
            if key in str(file_path):
                label = key
        img = read(file_path, 1)
        return img, label

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        imgs_1, classes = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img)[None] for img in imgs_1]
        labels = [torch.tensor([id2int[label]]) for label in classes]
        imgs, labels = [torch.cat(i).to(device) for i in [imgs, labels]]

        return imgs, labels, imgs_1


def conv_block(ni, no, kernel_size):
    return nn.Sequential(
        nn.Dropout(0.25),
        nn.Conv2d(ni, no, kernel_size=kernel_size, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2))


class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv_block(3, 64, 3),
            conv_block(64, 64, 3),
            conv_block(64, 128, 3),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 64, 3),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.Dropout(0.35),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, preds, targets):
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()
        return loss, acc


def train_per_epoch(train_dl, model, optimizer, criterion):
    """
    Per epoch training process
    :param train_dl: training set dataloader
    :param model: model to use
    :param optimizer: optimizer to be used
    :param criterion: loss functions for age and gender
    :return:
    train_loss: average loss per epoch
    """
    model.train()
    train_loss = []
    train_acc = []
    for batch_no, data in tqdm(enumerate(train_dl), total=len(train_dl), desc='Training',
                               unit='Batch', position=0, leave=True):
        # Unpack inputs and labels
        image, labels = data[0].to(device), data[1].to(device)
        # Zero gradients for each batch
        optimizer.zero_grad()
        # Predictions for this batch
        prediction = model(image)
        # Unpack and Compute loss
        loss, acc = criterion(prediction, labels)
        # Append batch loss and accuracy
        train_loss.append(loss.item())
        train_acc.append(acc.item())
        # Back-propagate loss
        loss.backward()
        # Change weights
        optimizer.step()
    # Calculate mean loss and accuracy
    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)
    return train_loss, train_acc


def evaluation(dl, model, desc):
    """
    Function containing the evaluation process of the model.
    :param dl: Dataloader object
    :param model: Model to be used
    :param desc: Whether the function is used during validation or testing
    :return:
    avg_v_loss: loss during evaluation for one epoch
    avg_v_acc: accuracy during evaluation for one epoch
    """
    model.eval()
    with torch.no_grad():
        validation_loss = []
        per_batch_val_acc = []
        # Iterate through validation data
        for _, vdata in tqdm(enumerate(dl), total=len(dl), desc=desc,
                             unit='Batch', position=0, leave=True):
            img, label = vdata[0].to(device), vdata[1].to(device)
            prediction = model(img)
            # Compute val accuracy, val loss
            v_loss, v_acc = model.compute_metrics(prediction, label)
            per_batch_val_acc.append(v_acc.item())
            validation_loss.append(v_loss.item())
        # Compute the average of each metric for one epoch
        avg_v_loss = np.mean(validation_loss)
        avg_vacc = np.mean(per_batch_val_acc)
    return avg_v_loss, avg_vacc


def early_stopping(model, filename, mode):
    """
    Function implementing early stopping techniques, using the mode variable.
    :param model: model to save
    :param filename: path and name of the file
    :param mode: whether to save the model or restore the best model from a path
    :return: NULL
    """
    if mode == 'save':
        torch.save(model.state_dict(), filename)
    elif mode == 'restore':
        model.load_state_dict(torch.load(filename))
    else:
        print("Not valid mode")


def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    """
    A simple function creating two plots to visualize accuracy
    and loss progression during training
    :param train_acc: list containing the training accuracy per epoch
    :param val_acc: list containing the validation accuracy per epoch
    :param train_loss: list containing the training loss per epoch
    :param val_loss: list containing the validation loss per epoch
    :return:
    """
    epochs = np.arange(1, len(train_acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flat
    ax[0].plot(epochs, train_loss, 'bo', label='Training loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation loss over increasing epochs')
    ax[1].plot(epochs, train_acc, 'bo', label='Training Accuracy')
    ax[1].plot(epochs, val_acc, 'r', label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation accuracy over increasing epochs')
    ax[1].legend()
    plt.grid('off')
    plt.show()


def display_predictions(model, labels, dataset, transforms):
    """
    A function to display randomly selected predictions
    :param model: Model to be used
    :param labels: dictionary containing class names
    :param dataset: dataset to extract predictions from
    :param transforms: transforms object to augment prediction images
    :return:
    """
    labels = list(labels.keys())
    figure, axs = plt.subplots(3, 5, figsize=(10, 8), constrained_layout=True)
    for i in range(15):
        # Randomly select and return an image and its label
        img, label = dataset.choose()
        original_img = img.copy()
        # Augment the image to be an appropriate input to the model
        img = transforms(img).to(device)
        predictions = model(img[None])
        # Save predictions to cpu and find the prediction
        predictions = predictions.to('cpu').detach().numpy()
        predicted_label = np.argmax(predictions)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # Display in the titles the predictions and actual labels
        if i <= 4:
            # Find the sign name and display it
            predicted_sign = labels[predicted_label]
            axs[0, i].imshow(original_img)
            axs[0, i].set_title('Prediction: {}'.format(predicted_sign),
                                color=("green" if predicted_sign == label else "red"))
            axs[0, i].set_xlabel('Actual: {}'.format(label))
        elif i <= 9:
            predicted_sign = labels[predicted_label]
            axs[1, i - 5].imshow(original_img)
            axs[1, i - 5].set_title('Prediction: {}'.format(predicted_sign),
                                    color=("green" if predicted_sign == label else "red"))
            axs[1, i - 5].set_xlabel('Actual: {}'.format(label))
        else:
            predicted_sign = labels[predicted_label]
            actual_sign = label
            axs[2, i - 10].imshow(original_img)
            axs[2, i - 10].set_title('Prediction: {}'.format(predicted_sign),
                                     color=("green" if predicted_sign == label else "red"))
            axs[2, i - 10].set_xlabel('Actual: {}'.format(label))

    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.show()


def prediction_metrics(model, test_dl, string_labels, confusion):
    """
    A function that uses test data to predict and provide the classification metrics
    and the confusion matrix
    :param model: Model to be used
    :param test_dl: Dataloader
    :param string_labels: dataframe column containing the labels in string format
    :param confusion: whether to plot confusion matrix
    :return:
    """
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        # Iterate through test data
        for _, test_data in tqdm(enumerate(test_dl), total=len(test_dl), desc='Prediction',
                                 unit='Batch', position=0, leave=True):
            img, label = test_data[0].to(device), test_data[1].to(device)
            prediction = model(img).cpu().numpy()
            # For each image, find the model prediction
            model_predictions = [np.argmax(arr) for arr in prediction]
            # Append to list the predictions and the ground truths
            y_pred.extend(model_predictions)
            y_true.extend(label.cpu().numpy())
    print(classification_report(y_pred=y_pred, y_true=y_true, target_names=list(string_labels), digits=3))
    # Check whether to plot confusion matrix
    if confusion:
        cf_matrix = confusion_matrix(y_true, y_pred)
        fig2, ax = plt.subplots(figsize=(12, 7))
        ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                               display_labels=string_labels).plot(xticks_rotation=-75, ax=ax)
        plt.title('Confusion Matrix')
        plt.show()


def im2gradcam(x, model):
    """
    Class Activation map function
    """
    im2fmap = nn.Sequential(*(list(model.model[:5].children()) + list(model.model[5][:2].children())))
    model.eval()
    logits = model(x)
    activations = im2fmap(x)
    pred = logits.max(-1)[-1]
    # get the model's prediction
    model.zero_grad()
    # compute gradients with respect to model's most confident logit
    logits[0, pred].backward(retain_graph=True)
    # get the gradients at the required feature map location
    # and take the avg gradient for every feature map
    pooled_grads = model.model[-6][1].weight.grad.data.mean((1, 2, 3))
    # multiply each activation map with corresponding gradient average
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]
    # take the mean of all weighted activation maps
    # (that has been weighted by avg. grad at each fmap)
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, 'Uninfected' if pred.item() else 'Parasitized'


def upsample_heatmap(map, img, sz):
    """
    Upsampling function to create the heatmap necessary for Class Activation Maps
    :param map: heatmap to upsample
    :param img: original image
    :param sz: input size
    :return: map: the upsampled map
    """
    m, M = map.min(), map.max()
    map = 255 * ((map - m) / (M - m))
    map = np.uint8(map)
    map = cv2.resize(map, (sz, sz))
    map = cv2.applyColorMap(255 - map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map * 0.7 + img * 0.3)
    return map
