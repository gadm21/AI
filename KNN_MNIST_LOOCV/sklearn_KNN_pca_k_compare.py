
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
 

pca_values = range(3, 300, 3)
k_values = range(1, 15)
results = []

for pca_value in pca_values :
    for k_value in k_values : 
        pca = decomposition.PCA(pca_value)
        pca.fit(flat_train_x) 
        pca_train_x = pca.transform(flat_train_x)
        pca_test_x = pca.transform(flat_test_x)  

        model = KNeighborsClassifier(n_neighbors=k_value)
        model.fit(pca_train_x, train_y)

        score = model.score(pca_test_x, test_y) 
        results.append((score, k_value, pca_value)) 