


import sys
import numpy as np
import random
from numpy.core.defchararray import translate

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint



from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, Activation, Concatenate
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.layers import LayerNormalization
# from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras import models
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import RMSprop, Adadelta
from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2


import tensorflow as tf
import tensorflow.keras.backend as K



import networkx as nx
import matplotlib.pyplot as plt


import time
import yaml
import random

translate_os = {
    'Windows': 0,
    'Linux': 1,
    'MacOs': 2,
    0 : 'Windows',
    1 : 'Linux',
    2 : 'MacOs'
}


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
    return None


class SimpleGraph():
  
    def __init__(self):
        self.V = 9
        self.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
            [4, 0, 8, 0, 0, 0, 0, 11, 0],
            [0, 8, 0, 7, 0, 4, 0, 0, 2],
            [0, 0, 7, 0, 9, 14, 0, 0, 0],
            [0, 0, 0, 9, 0, 10, 0, 0, 0],
            [0, 0, 4, 14, 10, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 1, 6],
            [8, 11, 0, 0, 0, 0, 1, 0, 7],
            [0, 0, 2, 0, 0, 0, 6, 7, 0]
            ]
        self.edges, self.weights = [], []
        self.get_edges_from_graph(self.graph)
        self.init_networkx_graph()
    
    def init_networkx_graph(self):
        self.G = nx.Graph()
        for i in range(self.V):
            self.G.add_node(i, os= 'no')
        for edge, weight in zip(self.edges, self.weights):
            self.G.add_edge(edge[0], edge[1], name = str(weight))
        self.pos = nx.spring_layout(self.G)
        self.edge_labels = nx.get_edge_attributes(self.G, 'name')
        self.node_labels = nx.get_node_attributes(self.G, 'os')
        
    
    def get_edges_from_graph(self, graph):
        for i in range(self.V):
            for j in range(self.V):
                if graph[i][j]:
                    self.edges.append((i, j))            
                    self.weights.append(graph[i][j])
    

    # A utility function to find the vertex with 
    # minimum distance value, from the set of vertices 
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
  
        # Initilaize minimum distance for next node
        min = 10_000
  
        # Search not nearest vertex not in the 
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
  
        return min_index
  
    # Funtion that implements Dijkstra's single source 
    # shortest path algorithm for a graph represented 
    # using adjacency matrix representation
    def dijkstra(self, src):
  
        dist = [10_000] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
  
        for cout in range(self.V):
  
            # Pick the minimum distance vertex from 
            # the set of vertices not yet processed. 
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
  
            # Put the minimum distance vertex in the 
            # shotest path tree
            sptSet[u] = True
  
            # Update dist value of the adjacent vertices 
            # of the picked vertex only if the current 
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and \
                dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
  
        return dist
  

    def draw(self, colors):
        nx.draw(self.G, pos = self.pos, node_color = colors, node_size = 500, with_labels = True)
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels)    
        # nx.draw_networkx_labels(self.G, self.pos, labels = self.node_labels)
        plt.show(block = False)
        plt.pause(0.01)

    def update(self, colors):
        plt.clf()

        self.draw(colors)



def get_random_node_props(os = 'Windows'):
    if isinstance(os, str): os = translate_os[os]
    ram = random.randint(1,10) * 1000
    network = random.randint(1, 100) * 100
    processor = random.randint(1, 5)
    cores = random.randint(1, 8)
    security = [random.randint(1, 20) for _ in range(3)]
    return [os, ram, network, processor, cores] + security


class Model:

    def __init__(self, yaml_path):

        K.clear_session()
        self.params = read_yaml(yaml_path)

        self.callbacks = []
        
        self.model = None
        if self.params['load_model'] :
            self.load_model()
        else:
            self.build_model()
            self.compile_model()


    def build_model(self):
        
        

        #__________________________Inputs
        inp = Input(shape = (9, 10))
        inp2 = Input(shape = (24,))





        #__________________________layers

        #____________Branch1 
        lstm_1_1 = LSTM(15, return_sequences = True)(inp)
        lstm_1_2 = LSTM(40, return_sequences = False)(lstm_1_1)
        d_1_3 = Dense(20, activation = 'relu')(lstm_1_2)

        #____________Branch2 
        d_2_1 = Dense(8, activation = 'relu')(inp2)

        #____________Concat 
        c_1 = Concatenate(1)([d_1_3, d_2_1])
        d_2 = Dense(20, activation = 'relu')(c_1)
        d_3 = Dense(9, activation = 'softmax')(d_2)
        
        
        self.model = models.Model(inputs = [inp, inp2], outputs = d_3)
        self.plot_model()
        print("model built")

    def load_model(self):
        assert self.params['model_path']
        print("loading model")
        self.model = models.load_model(self.params['model_path'])

    def compile_model(self, lr = None):
        if lr is None : 
            lr = 0.001

        self.model.compile(
            loss= 'categorical_crossentropy',
            optimizer= RMSprop(learning_rate = lr),
            metrics=["accuracy"]
        )
    
    def plot_model(self):
        tf.keras.utils.plot_model(self.model, 'light_model.png', show_shapes=True, show_layer_names=False) 
        tf.keras.utils.plot_model(self.model, 'specific_light_model.png', show_shapes=True) 
    

    def infer(self, nodes_info, application_requirements):
        assert self.model
        inputs1 = np.asarray([nodes_info], dtype = float)
        inputs2 = np.asarray([application_requirements], dtype = float)
        output = self.model.predict([inputs1, inputs2])
        return output



class AppMigrationEngine():

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.model = Model(yaml_path)
        self.graph = SimpleGraph()
        
        self.initialize_nodes()
        self.initialize_app_requirements()
        # self.app_node = 5

    def initialize_nodes(self):
        self.nodes = []
        for _ in range(3): self.nodes.append(get_random_node_props('Windows'))
        for _ in range(3): self.nodes.append(get_random_node_props('Linux'))
        for _ in range(3): self.nodes.append(get_random_node_props('MacOs'))

        for i, node in enumerate(self.nodes):
            self.graph.node_labels[i] = translate_os[node[0]]
        
        for i, node in enumerate(self.nodes):
            print("node:{} {},{}".format(i,translate_os[node[0]], node[1:]))

    def initialize_app_requirements(self):
        self.params = read_yaml(self.yaml_path)
        self.app_requirements, self.requirements_categories, self.requirements_types = [], [], []
        self.params['app_requirements']['os'][0] = translate_os[self.params['app_requirements']['os'][0]]
        for key in self.params['app_requirements'].keys():
            req = self.params['app_requirements'][key]
            self.app_requirements.append(req[0])
            self.requirements_categories.append(req[1])
            self.requirements_types.append(req[2])


    def recommend_nodes(self):

        def judge(cat_mask, type_mask, app_v, node_v):
          # see whether 
          score1 = 0
          for i in range(len(app_v)):
            if type_mask[i] == 1 : # mandatory
              if cat_mask[i] == 0 : # nominal
                if node_v[i] != app_v[i] : return 0
              else : # ordinal or numerical
                if node_v[i] < app_v[i] : return 0

            else: #optional
              if cat_mask[i] == 0 : # nominal
                if node_v[i] == app_v[i] : score1 += 1
                else: score1 -= 1
              else: # ordinal or numerical
                if node_v[i] >= app_v[i] : score1 += 1
                else: score1 -= 1

          return 1
          
        
        # migration_cost = self.graph.dijkstra(self.app_node)
        # for i, node in enumerate(self.nodes):
        #     print("node:{} os:{}".format(i, translate_os[node[0]]), end = ',')
        #     print(node[1:])
        # print("migration cost:", migration_cost)

        outputs = []
        app_node = None
        for i, node in enumerate(self.nodes):
            ret  = judge(self.requirements_categories, self.requirements_types, self.app_requirements, node)
            outputs.append(ret)
            if ret : app_node = i 
        # print("outputs:", outputs)
        # print("app node:", app_node)
        # print()
        # inputs1 = [(i, *self.nodes[i]) for i in range(self.graph.V)]
        # inputs2 = self.app_requirements+self.requirements_categories+self.requirements_types
        # outputs1 = self.model.infer(inputs1, inputs2)[0]
        # outputs1 = np.where(outputs1 > 0.5, 1, 0)
        # print("outputs:", outputs1)

        return outputs, app_node
        # outputs = np.argsort(outputs)[::-1]
        # return np.argmax(outputs)
        
        
    def update(self):
        self.initialize_app_requirements()
        recommendations, self.app_node = self.recommend_nodes()
        colors = ['blue'] * self.graph.V
        for i, node in enumerate(recommendations): 
            if node : colors[i] = 'red'
        if self.app_node :  colors[self.app_node] = 'green'
        
        # print("colors:", colors)
        self.graph.update(colors)



def main():

    params = read_yaml('vars.yaml')
    app = AppMigrationEngine('vars.yaml')

    while True:
        app.update()
        # time.sleep(0.1)


if __name__ == "__main__":
    main()



