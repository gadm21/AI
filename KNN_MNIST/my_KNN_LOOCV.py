
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
pca_test_x = list(pca.transform(flat_test_x)  )


k_values = np.arange(1,102,2)
all_predictions = KNN.LOOCV((pca_x, y), distance.euclidean2, k_values)


#visualize.pro_confusion_matrix(predictions, test_y)
#visualize.plot_confusion_matrix(matrix.astype(np.uint8))  