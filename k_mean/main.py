from utils import * 


image = read_image('china.png') 

data = image.reshape(-1, 3) 

print(data.shape) 

plot_data(data) 