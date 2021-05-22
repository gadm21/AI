
import threading
import queue
import cv2







def process_images(vid, q, lock):

    for i in range(50):
        lock.acquire()
        ret, frame = vid.read()
        lock.release()
        if not ret: continue 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q.put(("gray", gray, (20, 20)))
    
    q.put(("done", None, None))



def display_images(vid, q, lock):

    for i in range(50):
        lock.acquire()
        ret, frame = vid.read()
        lock.release()
        if not ret : continue 

        q.put(("raw", frame, (1000, 20)))

    q.put(("done", None, None))

