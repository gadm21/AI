import tensorflow as tf 
from utils import * 

def singleFlatten(image):
    return np.reshape(image, -1) 

(train, t_labels), (test, test_labels) = tf.keras.datasets.mnist.load_data()

print(test.shape) 


newtrain = [] 
newlabels = [] 

for i in range(test.shape[0]):
    newtrain.append(singleFlatten(test[i, : , : ]))
    newlabels.append(test_labels[i])


k_range = np.arange(1, 10, 2) 
KNN.LOOCV((newtrain, newlabels), distance.euclidean, k_range)
