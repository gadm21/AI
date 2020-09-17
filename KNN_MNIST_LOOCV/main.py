

from utils import *



train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"













if __name__ == "__main__":
    #train_x, train_y = reader.read(dir = train_path, labels_file = 'Training Labels.txt')
    test_x, test_y = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
    newtest_x, cleared = on_image.preprocess(test_x) 
    
    dataset = (newtest_x, test_y)
    k_range = np.arange(1,20, 2) 
    res = KNN.LOOCV(dataset, distance.euclidean, k_range) 
    
    r = np.arange(40, 60)
    for i in r : 
        print(res[i]) 
        on_image.show(cleared[i])
        print() 
        print() 