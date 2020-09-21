
from sklearn.neighbors import KNeighborsClassifier 
from utils import * 
from sklearn import decomposition
from sklearn.metrics import classification_report

train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"



test_x, test_y = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
flat_test_x = on_image.preprocess(test_x)

train_x, train_y = reader.read(dir = train_path, labels_file = 'Training Labels.txt')
flat_train_x = on_image.preprocess(train_x) 
 
samples = []
labels = []
for i in range(0, 100, 20):
    samples+= flat_test_x[i:i+10]
    labels+= test_y[i:i+10]

pca = decomposition.PCA(3)
pca.fit(flat_train_x) 

pca_test_x = list(pca.transform(samples)) 
print(len(pca_test_x))
print(pca_test_x[0].shape)

visualize.plot3D(pca_test_x, labels)