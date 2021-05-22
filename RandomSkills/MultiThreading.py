

from utils import *


alive_threads = 2 
q = queue.Queue()
lock = threading.Lock()
vid = cv2.VideoCapture(0)

threading.Thread(target = process_images, args = (vid, q, lock)).start()
threading.Thread(target = display_images, args = (vid, q, lock)).start()


while alive_threads :
    try:
        name, image, loc = q.get()
        if name == "done" : alive_threads -= 1
        else:
            cv2.namedWindow(name)
            cv2.moveWindow(name, loc[0], loc[1])
            cv2.imshow(name, image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    except:
        pass


cv2.destroyAllWindows()