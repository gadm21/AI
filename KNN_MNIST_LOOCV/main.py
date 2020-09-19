

from utils import *



train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"













if __name__ == "__main__":
    
    images, labels = reader.read(dir = test_path, labels_file = 'Test Labels.txt')

    for i in range(10):
        print(labels[i])
        on_image.show(images[i])