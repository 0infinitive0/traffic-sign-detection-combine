from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pytube import YouTube
import os
import cv2
import numpy as np
from ultralytics import YOLO
import math
import torch
from keras.models import load_model

heights = 32
widths = 32


def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def normaly(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (heights, widths))
    #plt.imshow(img, cmap = plt.get_cmap('gray'))
    #plt.show()
    img = img.reshape(1, 32, 32, 3)
    img = img / 255
    return img
def relu(x):
    return x * (x > 0)
def pytago(width,height):
    return math.floor(math.sqrt(width * width + height * height))


app = Flask(__name__)

# Set the path for uploaded videos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ano = ["Do not enter","No stopping or parking","No parking","Maximum speed 40 km/h","Maximum speed 50 km/h","Maximum speed 60 km/h","Maximum speed 70 km/h","Maximum speed 80 km/h","No right turn","No left turn","Keep right","No 2 & 3 wheel vehicles","Roundabout ahead","No cars","No U-turn","No buses","Bus stop","No motorcycles","Height restriction","No heavy vehicles","Left road junction with priority","Right road junction with priority","Zebra crossing / crosswalk ahead","Traffic obstruction ahead - may pass on either side","Slow down","School zone ahead","Road narrows ahead on the left side","No trailers","Hospital nearby","null"]

def process_video(video_path):
    model1 = YOLO('Manual yolov8n2.pt')  # load detection model
    model2 = load_model('manual LeNet4 (no pre).h5')  # load classification model

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

    trackers = cv2.legacy.MultiTracker_create()
    label = np.array([], dtype=object)
    fnum = 0  # frame .no
    avgfps = 0
    sampling_rate = 5
    cap = cv2.VideoCapture(video_path)

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
                if con < 0.50:
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
                label = np.append(label, str(ano[int(la)]))
                
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

@app.route('/')
def index():
    return render_template('index_video.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        process_video(file_path)

    return render_template('index_video.html', video_path=file_path)


@app.route('/youtube', methods=['POST'])
def youtube():
    video_url = request.form['video_url']

    if video_url:
        yt = YouTube(video_url)
        video = yt.streams.get_highest_resolution()
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'youtube_video.mp4')
        video.download(app.config['UPLOAD_FOLDER'],
                       filename='youtube_video.mp4')


        process_video(file_path)  # Call the process_video function
    return render_template('index_video.html', video_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)
