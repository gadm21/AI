from utils import * 


path = 'plates/1.png' 
image = load_image(path) 

edge = Canny_detector(image) 

show_image(edge) 