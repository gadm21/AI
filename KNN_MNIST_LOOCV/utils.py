import os 
import cv2 
import numpy as np 

class reader :
    def read_data( path, to_gray = True):
        names = os.listdir(path)
        data = [] 
        sizes = []
        for name in names :
            if name.endswith('txt') : continue 
            image = cv2.imread(os.path.join(path, name)) 
            if to_gray : image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            data.append(image)  
        return data

    def read_labels( path):
        raw = ""
        with open(path) as file:
            raw = file.read()
        return list(map(int, raw.split()))

    def read(dir, labels_file):
        data =  reader.read_data(dir) 
        labels = reader.read_labels(os.path.join(dir, labels_file))  
        return data, labels


class on_image :

    def show( image, label = 'r'):
        cv2.imshow(label, image)
        cv2.waitKey(0) 
        cv2.destroyWindow(label) 
    
    def flatten(images):
        return [np.reshape(image, -1) for image in images]
        

    def binarize(images, threshold = 200):
        return [np.where(image<=threshold, 0, 255).astype(np.uint8) for image in images]


    def clear(images, k = 3):
        
        result = [] 

        def check(p):
            return p.sum() > (255*p.shape[0] )

        for image in images :
            newimage = np.zeros(image.shape) 
            for row in range(k//2, image.shape[0] - k//2):
                for col in range(k//2, image.shape[1] - k//2):
                    if check(image[ row-(k//2):row+(k//2)+1, col-(k//2):col+(k//2)+1]):
                        newimage[ row, col] = 255
            result.append(newimage) 
        
        return result

    def preprocess(images):
        binarized = on_image.binarize(images) 
        cleared = on_image.clear(binarized) 
        flattened = on_image.flatten(cleared) 
        return flattened, cleared


class distance:

    def euclidean2(v1, v2):
        dist = 0 
        for p1, p2 in zip(v1, v2):
            dist += (p1-p2)**2
        return np.sqrt(dist) 

    def euclidean(v1, v2):
        return np.linalg.norm((v1-v2))

class KNN:

    def count_votes(votes):
        counts = dict() 
        for vote in votes : 
            if counts.get(vote): counts[vote] += 1
            else: counts[vote] = 0 
        counts = sorted(counts, key = lambda item : item[1], reverse = True)  
        return counts[0][0] 

    def LOOCV(dataset, dist_func, k_range):
        data = dataset[0] 
        labels = dataset[1] 



        all_distances = [] 
        for i in range(len(data)):
            target = data[i]
            distances = [(label, dist_func(target, newdata)) for label, newdata in zip(labels, data[:i] + data[i+1:])]
            distances = sorted(distances, key = lambda item : item[1])
            all_distances.append(distances) 
         
        accuracies = [] 
        for k in k_range:
            accuracy = sum([labels[i]==KNN.count_votes(all_distances[i][:k]) for i in range(len(data))]) / len(labels) * 100
            accuracies.append(accuracy) 
            print(accuracy)
        return all_distances

    def predict(dataset, target, k, dist_func):
        data = dataset[0] 
        labels = dataset[1] 
        distances = [(label, dist_func(target, x)) for label, x in zip(labels, data)]
        distances = sorted(distances, key = lambda item : item[1])[:k] 
        result = KNN.count_votes(distances) 
        return result 
