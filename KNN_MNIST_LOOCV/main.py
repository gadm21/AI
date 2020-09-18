

from utils import *



train_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Train"
test_path = r"C:\Users\gad\Downloads\Computer RA Task\Computer RA Task\Task Dataset\Test"













if __name__ == "__main__":
    
    images, labels = reader.read(dir = test_path, labels_file = 'Test Labels.txt')
    images = on_image.median(images) 
    
    no_image.show(images[0])