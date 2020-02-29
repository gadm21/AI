
import numpy as np 
import config 

def reshape_and_onehot(images, labels):
    images= np.reshape(images, [images.shape[0], images.shape[1], images.shape[2], 1]).astype(np.float32)
    onehot_labels= np.zeros((labels.shape[0], config.num_classes))
    onehot_labels[np.arange(labels.shape[0]), labels] = 1

    return images, onehot_labels


def randomize(images, labels):
    permutation = np.random.permutation(labels.shape[0])
    images= images[permutation, :, :, :]
    labes= labels[permutation]

    return images, labels

def get_next(images, labels, start, end):
    images_subset= images[start:end]
    labels_subset= labels[start:end]

    return images_subset, labels_subset