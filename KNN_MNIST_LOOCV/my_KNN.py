
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import decomposition
from utils import * 
from sklearn.metrics import classification_report

train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"



test_x, test_y = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
flat_test_x = on_image.preprocess(test_x)

train_x, train_y = reader.read(dir = train_path, labels_file = 'Training Labels.txt')
flat_train_x = on_image.preprocess(train_x) 
 



pca = decomposition.PCA(50)
pca.fit(flat_train_x) 
pca_train_x = pca.transform(flat_train_x)
pca_test_x = pca.transform(flat_test_x)  

k_values = np.arange(1,7)
KNN.LOOCV((pca_train_x, train_y), distance.euclidean2, k_values)