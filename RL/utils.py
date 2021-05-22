



import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle
import time
from matplotlib import style
import random
import os


style.use("ggplot") 


SIZE = 10
allow_show = True

action_space = 4
episodes = 10000

move_penalty = -1
enemy_penalty = -300
food_reward = 50

learning_rate = 0.1
epsilon = 0.01
eps_decay = 0.9999
discount = 0.95
show_every = np.maximum(10, episodes//20)


player_n = 1
food_n = 1
enemy_n = 1

colors = {
    "player" : [(255, 0, 0)],
    "food" : [(0, 255, 0)],
    "enemy" : [(0, 0, 255)]
}








class Blob(object):

    def __init__(self, color):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        self.color = color
    
    def __str__(self):
        return "x:{} y:{}".format(self.x, self.y)
    
    def __sub__(self, other):
        return (self.x-other.x , self.y-other.y)
    
    def loc(self):
        return (self.x, self.y)

    def action(self, choice):
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = 1)
        elif choice == 2:
            self.move(x = 1, y = -1)
        elif choice == 3:
            self.move(x = -1, y = -1)
        
    def move(self, x = False, y = False):
        self.x += x if x else np.random.randint(-1, 2)
        self.y += y if y else np.random.randint(-1, 2)

        # if the player is out of bound
        self.x = np.maximum(0, np.minimum(SIZE-1, self.x))
        self.y = np.maximum(0, np.minimum(SIZE-1, self.y))
    
    def coincide(self, other):
        return self.x == other.x and self.y == other.y
    


def get_q_table(path = None):

    if path is None or not os.path.isfile(path):
        print("creating the Q table")
        q_table = {}
        for i in range(-SIZE+1, SIZE):
            for j in range(-SIZE+1, SIZE):
                for k in range(-SIZE+1, SIZE):
                    for l in range(-SIZE+1, SIZE):
                        q_table[((i,j),(k,l))] = [np.random.uniform(-5, 0) for m in range(action_space)]
    else:
        print("loading the Q table")
        with open(path, 'rb') as f:
            q_table = pickle.load(f)
        
    return q_table


def render(blobs, grid_size, to_size = 500):
    env = np.zeros((grid_size, grid_size, 3), dtype = np.uint8)
    for blob in blobs.values():
        env[blob.loc()] = blob.color
    
    image = Image.fromarray(env, "RGB")
    image = image.resize((500, 500))

    return np.array(image)


def show_image(image, period = 60):
    cv2.imshow("env", image)
    if cv2.waitKey(period) & 0xFF == ord('q'):
        pass
    # cv2.destroyWindow("image")



def get_reward(blobs):
    player = blobs["player"]
    enemy = blobs["enemy"]
    food = blobs["food"]

    if player.coincide(enemy):
        reward = enemy_penalty
    elif player.coincide(food):
        reward = food_reward
    else:
        reward = move_penalty

    return reward



def get_observation(blobs):
    player = blobs["player"]
    enemy = blobs["enemy"]
    food = blobs["food"]

    return (player-food, player-enemy)



def get_action(q_table, observation):
    if np.random.random() < epsilon:
        action = np.random.randint(0, action_space)
    else:
        action = np.argmax(q_table[observation])
    
    return action
    


def update_q_table(q_table, blobs, observation, action, reward):
    new_observation = get_observation(blobs)
    current_q = q_table[observation][action]
    max_new_q = np.max(q_table[new_observation])

    if reward == food_reward or reward == enemy_penalty:
        new_q = reward
    else: new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_new_q)
    q_table[observation][action] = new_q


def save_q_table(q_table, path):
    with open(path, 'wb') as f :
        pickle.dump(q_table, f)



def save_plt(signal, name = "signal"):
    fig = figure(figsize=(20, 10))
    plt.plot([i for i in range(len(signal))], signal)
    plt.ylabel("reward {}ma".format(show_every))
    plt.xlabel("episode")
    fig.savefig(name + ".png")
    plt.close()