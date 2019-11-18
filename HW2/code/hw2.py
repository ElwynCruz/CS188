import numpy as np
import cv2
from skimage.feature import match_template
from skimage import io
import matplotlib.pyplot as plt
import math

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

            #cv2.imshow('frame',gray)
            img_path = '../frames/frame'+ str(i) + '.jpg'
            cv2.imwrite(img_path, gray)
            img = cv2.imread(img_path, 0)
            frames.append(img)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    return frames

def findTemplate(images):
    template = cv2.imread('../img/template.jpg', 0)

    # ij = np.unravel_index(np.argmax(result), result.shape)
    # x, y = ij[::-1]
    # fig = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(1, 3, 2)

    # ax.imshow(image, cmap=plt.cm.gray)
    # ax.set_axis_off()

    # htemplate, wtemplate = template.shape
    # rect = plt.Rectangle((x, y), wtemplate, htemplate, edgecolor='r',
    #  facecolor='none', label='Template')
    # ax.add_patch(rect)
    # plt.legend()
    # plt.show()

    results = []
    for i in images:
        result = match_template(i,template,pad_input=True)
        #o.imshow(result, cmap='gray')
        #plt.show()
        results.append(result)
    return results

def plotMaximas(matches):
    x = []
    y = []
    for i in matches:
        loc = np.argmax(i)
        x.append(math.floor(loc / len(i[0])))
        y.append(loc % len(i[0]))
    plt.plot(y,x)
    plt.show()

def defocusImage(matches, img):
	x = []
	y = []
	transformation_matrix = []
	for i in matches:
		loc = np.argmax(i)
		# x = rows
		x.append(math.floor(loc / len(i[0])))
		#y = columns
		y.append(loc % len(i[0]))
	for i in range(x):
		shift = (x,y)
		transformation_matrix.append(shift)
	


if __name__ == '__main__':
    videoStr = "../img/video.MOV"
    frames = extractFrames(videoStr)
    matchedframes = findTemplate(frames)
    defocusImage(matchedframes, frames[0])
    #plotMaximas(matchedframes)