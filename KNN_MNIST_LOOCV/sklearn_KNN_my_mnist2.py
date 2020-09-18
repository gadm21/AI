
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import decomposition
from utils import * 
from sklearn.metrics import classification_report

train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"



test_x, test_y = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
flat_test_x, clear_test = on_image.preprocess(test_x)

train_x, train_y = reader.read(dir = train_path, labels_file = 'Training Labels.txt')
flat_train_x, clear_train = on_image.preprocess(train_x) 
 

pca_components = 90
pca = decomposition.PCA(pca_components)
pca.fit(flat_train_x) 
pca_train_x = pca.transform(flat_train_x)
pca_test_x = pca.transform(flat_test_x)  


k = 7 
model = KNeighborsClassifier(n_neighbors=k)
model.fit(pca_train_x, train_y)

score = model.score(pca_test_x, test_y)  
print("score:", score)
