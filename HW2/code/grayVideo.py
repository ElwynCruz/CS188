import numpy as np
import cv2


def extractFrames(video_name):
    cap = cv2.VideoCapture(video_name)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    i=0
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame',gray)
            img_path = '../frames/video'+ str(i) + '.jpg'
            cv2.imwrite(img_path, gray)
            img = cv2.imread(img_path)
            frames.append(img_path)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    return frames
