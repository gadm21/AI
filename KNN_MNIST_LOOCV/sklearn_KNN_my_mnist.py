
from sklearn.neighbors import KNeighborsClassifier 
from utils import * 
from sklearn.metrics import classification_report

train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"



test_x, test_y = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
flat_test_x, clear_test = on_image.preprocess(test_x)

train_x, train_y = reader.read(dir = train_path, labels_file = 'Training Labels.txt')
flat_train_x, clear_train = on_image.preprocess(train_x) 
 

print("training points:", len(train_y)) 
print("testing points:", len(test_y)) 


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(flat_train_x, train_y)

	# evaluate the model and update the accuracies list
	score = model.score(flat_test_x, test_y)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))


model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(flat_train_x, train_y)
predictions = model.predict(flat_test_x)
 
# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(test_y, predictions))

