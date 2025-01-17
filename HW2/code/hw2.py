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
    gray_frames = []
    color_frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            color_path = '../frames/color'+ str(i) + '.jpg'
            gray_path = '../frames/gray'+ str(i) + '.jpg'
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cv2.imshow('frame',gray)
            cv2.imwrite(color_path, frame)
            color_img = cv2.imread(color_path)
            color_frames.append(color_img)

            cv2.imwrite(gray_path, gray)
            gray_img = cv2.imread(gray_path, 0)
            gray_frames.append(gray_img)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    return gray_frames, color_frames

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
    shift = []
    for i in matches:
        loc = np.argmax(i)
        x2 = math.floor(loc / len(i[0]))
        y2 = loc % len(i[0])
        xy = []
        xy.append(x2)
        xy.append(y2)
        shift.append(xy)
    #plt.plot(y,x)
    # plt.show()
    shifts = np.array(shift)
    shifts = shifts - shifts[0]
    return shifts

# def defocusImage(matches, colorframes):

def defocusedImage(shifts, colorframes):
    firstFrame = colorframes[0]
    finalImage = np.int32(colorframes[0])
    
    for num,i in enumerate(shifts):
        transMat = np.float32([[1,0,-i[1]],[0,1,-i[0]]])
        rows, cols, intensity = np.shape(firstFrame)
        transImg = cv2.warpAffine(colorframes[num], transMat, (cols,rows))
        finalImage += np.array(np.int32(transImg))
    finalImage = finalImage / len(shifts)
    finalImage = np.uint8(finalImage)
    cv2.imwrite('../test.jpg', finalImage)
	
def plotFunc():
    # w = []
    # d = []
    # for i in range(100):
    #     if i == 0:
    #         continue
    #     for j in range(100):
    #         if j == 0:
    #             continue
    #         diff = abs(i-j)
    #         width = float(diff)/(i*j)
    #         d.append(diff)
    #         w.append(width)
    # plt.scatter(d,w)
    # plt.show()     
    w = []
    f = []
    for i in range(100):
        w.append(i)
        f.append(i*.5)
    plt.scatter(f,w)
    plt.show()      

if __name__ == '__main__':
    videoStr = "../img/video.MOV"
    grayframes, colorframes = extractFrames(videoStr)
    matchedframes = np.array(findTemplate(grayframes))
    shifts = plotMaximas(matchedframes)
    # plotMaximas(matchedframes)
    defocusedImage(shifts,colorframes)
    plotFunc()