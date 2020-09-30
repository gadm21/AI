
from sklearn import decomposition
from utils import * 

train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"



test_x, test_y = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
flat_test_x = on_image.preprocess(test_x)

x, y = reader.read(dir = train_path, labels_file = 'Training Labels.txt')
flat_x = on_image.preprocess(x) 
 



pca = decomposition.PCA(42)
pca.fit(flat_x) 
pca_x = list(pca.transform(flat_x))
pca_test_x = list(pca.transform(flat_test_x))




evaluate = KNN.classifier(1, pca_x, y, distance.euclidean)
score = evaluate(pca_test_x, test_y)

print("score:", score) 