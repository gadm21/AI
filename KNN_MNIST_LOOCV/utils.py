import os 
import cv2 
import numpy as np 
import pandas as pd
import random 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn

class reader :
    def read_data( path, names, to_gray = True): 
        data = [] 
        sizes = []
        for name in names : 
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
        if 'Train' in labels_file : names = ['N'+str(i)+'.jpg' for i in range(1,2401)]
        else : names = ['N'+str(i)+'.jpg' for i in range(1, 201)]

        data =  reader.read_data(dir, names) 
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
            return p.sum() > (0.5 *255*p.shape[0]*p.shape[1] )

        for image in images :
            newimage = np.zeros(image.shape, dtype = np.uint8) 
            for row in range(k//2, image.shape[0] - k//2):
                for col in range(k//2, image.shape[1] - k//2):
                    if check(image[ row-(k//2):row+(k//2)+1, col-(k//2):col+(k//2)+1]):
                        newimage[ row, col] = 255
            result.append(newimage) 
        
        return result

    def preprocess(images):
        medianed = on_image.median(images) 
        #binarized = on_image.binarize(medianed, threshold = 100)
        cropped = on_image.crop(medianed)
        flattened = on_image.flatten(cropped) 
        return flattened 

    def crop(images):

        newimages= []
        for image in images : 
            nz_y, nz_x= np.nonzero(image)[0], np.nonzero(image)[1]
            newimage = image[min(nz_y): max(nz_y),min(nz_x):max(nz_y)]
            newimage = cv2.resize(newimage, image.shape).astype(np.uint8)
            newimages.append(newimage) 
        return newimages

    def median(images, k=3):
        def med(p):
            newp = np.sort(np.reshape(p, p.shape[0]*p.shape[1]))
            return newp[newp.shape[0]//2]

        newimages = [] 
        for image in images:
            newimage = np.zeros(image.shape, dtype = np.uint8) 
            for row in range(k//2, image.shape[0]-k//2):
                for col in range(k//2, image.shape[1]-k//2):
                    newimage[row, col] = med(image[row-(k//2):row+(k//2)+1, col-(k//2):col+(k//2)+1])
            newimages.append(newimage) 
        return newimages

    def show_many(many_images):
        all = np.hstack(many_images) 
        on_image.show(all)


class distance:

    def euclidean2(v1, v2): 
        total_squared_dist = np.sum((v1-v2)**2)
        return np.sqrt(total_squared_dist) 

    def euclidean(v1, v2):
        return np.linalg.norm((v1-v2))

    def MAE(v1, v2):
        return np.sum(np.abs(v1-v2)) / v1.shape[0]
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
        sz = len(data)

        all_distances = [] 
        for i in range(sz):
            target = data[i]
            remaining = data[:i]+ data[i+1:]  
            distances = [(label, dist_func(target, newdata)) for label, newdata in zip(labels, remaining)]
            distances = sorted(distances, key = lambda item : item[1])
            all_distances.append(distances) 
         
        accuracies = [] 
        for k in k_range:
            predictions = [KNN.count_votes(all_distances[i][:k]) for i in range(sz)]
            accuracy = sum([labels[i]==KNN.count_votes(all_distances[i][:k]) for i in range(sz)]) / sz * 100
            accuracies.append(accuracy) 
            print("k value:{} accuracy:{}".format(k, accuracy) )
            return KNN.my_confusion_matrix(predictions, labels), predictions
        

    def predict(dataset, target, k, dist_func):
        data = dataset[0] 
        labels = dataset[1] 
        distances = [(label, dist_func(target, x)) for label, x in zip(labels, data)]
        distances = sorted(distances, key = lambda item : item[1])[:k] 
        result = KNN.count_votes(distances) 
        return result 


    def my_confusion_matrix(predictions, labels):
        size = len(np.unique(labels)) + 1 #add one to write labels, up and left, in the confusion matrix picture
        cm = np.zeros((size, size), dtype = np.uint8) 
        for i in range(len(labels)):
            cm[labels[i]+1, predictions[i]+1] += 1

        return cm 




class visualize:
    def plot_my_confusion_matrix(cm):
        viewable = ((cm - np.min(cm)) / (np.max(cm) - np.min(cm)) ) * 255
        viewable = viewable.astype(np.uint8)  
        viewable = cv2.resize(viewable, (500,500), interpolation = cv2.INTER_NEAREST)

        for i in range(2, 12):
            offset = i * 45
            viewable = cv2.putText(viewable, str(i-2), (offset - 22,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
            viewable = cv2.putText(viewable, str(i-2), (10,offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        on_image.show(viewable) 

    def plot3D(data, labels):

        fig = plt.figure(1, figsize = (10,7))
        ax = Axes3D(fig, rect = [0,0,1,1], elev =48, azim = 134)

        xs, ys, zs = [], [], [] 
        for d in data : 
            xs.append(d[0])
            ys.append(d[1]) 
            zs.append(d[2])

        
        ax.scatter(xs, ys, zs, c=labels)

        plt.show()


    def pro_confusion_matrix(predictions, labels):
        pd_labels = pd.Series(labels, name='true') 
        pd_pred = pd.Series(predictions, name='predictions')
        cm = pd.DataFrame(pd.crosstab(pd_labels, pd_pred))

        plt.figure() 
        sn.heatmap(cm, annot= True)

        plt.show()
