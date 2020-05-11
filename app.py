import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import pickle
import time
import os
import json 
import time
from imutils import paths
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from flask import Flask, render_template, Response,request



###################flask area######################################
app = Flask(__name__)




@app.route("/")
def index():
    """Video streaming home page."""
    return render_template('index.html')
def gen():
    
          
    from imutils.video import VideoStream
    from imutils.video import FPS
    import numpy as np
    import argparse
    import pickle
    import time
    import os
    import json 
    import time
    from imutils import paths
    import numpy as np
    import argparse
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    ############################################################TAKING USER CREDENTIAL####################################################
    #print("enter name")
  
    x = "Authorised"
  
     
    
    # Directory 
    #directory = "{}".format(x)

    user_name=x

    
    count=300
    """

    # Parent Directory path 
    parent_dir = "dataset"
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path) 
    print("Directory '% s' created" % directory) 
    """

    vs = VideoStream(src=0).start()
    fps = FPS().start()
    img_counter = 0

    while True:
        frame = vs.read()
        img_name = "dataset/{}/0000{}.png".format(x,img_counter)
        cv2.imwrite(img_name, frame)
        #print("{} written!".format(img_name))
        img_counter += 1
        if img_counter == count:
            break

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # if the `q` key was pressed, break from the loop
        
    fps.stop()        
    cv2.destroyAllWindows()
    vs.stop()

    print("Taken ----{}-------images".format(count))
    ###################################################################################DETCTION MODEL#######################################################
    print("DETECTION MODEL START")
    # import the necessary packages


    ######################################extract_embeddings########################################################



    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    args["detector"]="face_detection_model"
    args["dataset"]="dataset"
    args["embedding_model"]="openface_nn4.small2.v1.t7"
    args["embeddings"]="output/embeddings.pickle"


    print("--------------------------------extracting_embeddings-----------------------------------")



    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0
    print("Extracting Features---------->>>")
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        import imutils
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(args["embeddings"], "wb")
    f.write(pickle.dumps(data))
    f.close()





    print("--------------------------TRAINING MODEL-----------------------")
    ####################################################################################################33


    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=False,
        help="path to serialized db of facial embeddings")
    ap.add_argument("-r", "--recognizer", required=False,
        help="path to output model trained to recognize faces")
    ap.add_argument("-l", "--le", required=False,
        help="path to output label encoder")
    args = vars(ap.parse_args())


    args["embeddings"]="output/embeddings.pickle"
    args["recognizer"]="output/recognizer.pickle"
    args["le"]="output/le.pickle"




    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(args["embeddings"], "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(args["recognizer"], "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(args["le"], "wb")
    f.write(pickle.dumps(le))
    f.close()


    #########################################################################################################

    print("--------------------------------RUNNING MAIN RECOGNITION FILE--------------------")


    #########################################yolo space################################################################
    phone_counter=0
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    args["yolo"]="yolo-coco"




    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
    ####################################################################################################



    start_time = time.time()
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())


    args["detector"]="face_detection_model"
    args["embedding_model"]="openface_nn4.small2.v1.t7"
    args["recognizer"]="output/recognizer.pickle"
    args["le"]="output/le.pickle"


    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    ########################################################
    ######################################################


    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # start the FPS throughput estimator
    fps = FPS().start()
    flag=0
    flag_count=1
    # loop over frames from the video file stream
    while True:
        dictionary ={ 
        "Time_Stamp" : 0, 
        "Type" : "Authorised Person", 
        "Description" : "Authorised Person is present"
        }
        f=0
        g=0    
        # grab the frame from the threaded video stream
        frame = vs.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        end_time = time.time()
        dictionary["Time_Stamp"]=end_time-start_time
    #####################################phone with yolo #############################################################################  
        # read the next frame from the file

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)          
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                if LABELS[classIDs[i]] == "cell phone":
                    phone_counter=phone_counter+1;
                    img = "output/0000{}.png".format(phone_counter)
                    cv2.imwrite(img, frame)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])

                    cv2.putText(frame, "PHONE DETECTED", (200,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)






    ######################################################################################################################################3333

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.7:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                if name ==user_name:
                    name="CORRECT USER"
                # draw the bounding box of the face along with the
                # associated probability
                text = name
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                if text=="unknown":
                    cv2.putText(frame, "WARNING: UNAUTHORISED PERSON DETECTED", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    g=1
                    flag=flag+1
                if flag>100:
                    img_name = "default/0000{}.png".format(flag_count)
                    cv2.imwrite(img_name, frame)
                    flag_count=flag_count+1
                    time.sleep(0.5)
                    flag=0


                # Serializing json  
        if f==1:
            dictionary["Type"]="Phone"
            dictionary["Description"]="User is using phone"

        if g==1:
            dictionary["Type"]="Unauthorised Person"
            dictionary["Description"]="MULTIPLE PERSONS ARE DETECTED"
        if g==1 and f==1:
            dictionary["Type"]="Phone AND Authorised"
            dictionary["Description"]="User is using hin phone AND Authorised Person is present"


        json_object = json.dumps(dictionary, indent = 4) 

    # Writing to sample.json 
        with open("sample.json", "a+") as outfile: 
            outfile.write(json_object) 

        # update the FPS counter
        fps.update()

        # show the output frame
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
   
        

# stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()






        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an frame tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
    app.debug = True
















