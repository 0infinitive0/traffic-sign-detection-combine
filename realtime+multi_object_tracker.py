import cv2
import  numpy as np
from ultralytics import YOLO
import math
import cv2
import torch
from keras.models import load_model

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def normaly(img):
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    #plt.imshow(img, cmap = plt.get_cmap('gray'))
    #plt.show()
    img = img.reshape(1, 32, 32, 1)
    return img
def relu(x):
    return x * (x > 0)
def pytago(width,height):
    return math.floor(math.sqrt(width * width + height * height))



cap = cv2.VideoCapture('Germany_480p.mp4')  #from video
# cap = cv2.VideoCapture(0)  #from camera
model1 = YOLO('Yolov8n.pt') #load detection model
model2 = load_model('LeNet.h5') #load classification model


# create a dictionary of all trackers in OpenCV that can be used for tracking
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}


# Create MultiTracker object
trackers = cv2.legacy.MultiTracker_create()
label = np.array([],dtype=object)
fnum = 0    #frame .no
avgfps = 0
sampling_rate = 5

while True:
    timer = cv2.getTickCount()  #for fps caculating
    frame = cap.read()[1]
    fnum += 1

    if (fnum%sampling_rate) == 0:
        #run detection model
        results = model1(frame,verbose=False)
        results1 = results[0].boxes
        ima = np.array(frame)
        #create new multi tracker
        trackers = cv2.legacy.MultiTracker_create()
        #clear existing label list
        label = np.array([], dtype=object)
        #get result from detection model
        for bo in results1:
            row = bo.xyxy[0].type(torch.int64).tolist()
            con = round(float(bo.conf[0]), 2)
            #if detection confident lower than 80, skip
            if con < 0.80:
                continue
            xmin = relu(row[0] - 5)
            ymin = relu(row[1] - 5)
            xmax = relu(row[2] + 5)
            ymax = relu(row[3] + 5)
            # print(xmin,ymin,xmax,ymax)
            cropped_image = ima[ymin:ymax, xmin:xmax]
            #         plt.imshow(cropped_image)
            #         plt.show()
            width = row[2] - row[0]
            height = row[3] - row[1]
            #add bounding box to tracker
            roi = [row[0],row[1],row[2] - row[0],row[3] - row[1]]
            #preprocess image
            im = normaly(cropped_image)
            # run classification model
            y_pred1 = model2.predict(im, verbose=0)
            #predict label
            la = int(np.argmax(y_pred1, axis=1))
            #predict propability
            prop = round(y_pred1[0][la], 2)
            #print(type(row))
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            tracker = OPENCV_OBJECT_TRACKERS['kcf']()
            trackers.add(tracker, frame, tuple(roi))
            #add label to list
            label = np.append(label, str(la+1))

    if frame is None:
        break
    #frame = cv2.resize(frame,(1090,600))

    (success, boxes) = trackers.update(frame)
    #print(success,boxes)
    # loop over the bounding boxes and draw then on the frame
    #print(label)
    if success == False:
        bound_boxes = trackers.getObjects()
        idx = np.where(bound_boxes.sum(axis= 1) != 0)[0]
        bound_boxes = bound_boxes[idx]
        label = label[idx]
        #print(type(bound_boxes))
        trackers = cv2.legacy.MultiTracker_create()
        for bound_box in bound_boxes:
            trackers.add(tracker,frame,bound_box)
        continue
    #put bounding box
    for i,box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,label[i],(x+10,y-3),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)
    k = cv2.waitKey(1)
    #caculate fps
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if fnum == 1:
        avgfps = fps
    else:
        avgfps = (avgfps+fps)/2

    cv2.putText(frame, 'fps: ' + str(int(avgfps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Frame', frame)

    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


cap.release()
cv2.destroyAllWindows()
