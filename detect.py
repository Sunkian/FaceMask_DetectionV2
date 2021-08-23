import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import paho.mqtt.client as mqtt
import io
import json
import base64
import string
import random
from PIL import Image
from time import time
from subprocess import Popen, PIPE
from utils.datasets import LoadStreams
# import psutil

config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[
             :2]  # shape is (height × width × other dimensions) then we can grab just height and width by taking the first two elements, [:2], and unpacking them appropriately.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def detect(source, output, mqtt_address, mqtt_topic, rtsp_transport='tcp'):
    print()
    print("INPUT RTSP URL\t-\t", source if len(source) > 0 else "None")
    print("OUTPUT RTSP URL\t-\t", output if len(output) > 0 else "None")
    print("MQTT URL\t-\t", mqtt_address if len(mqtt_address) > 0 else "None")
    print("MQTT TOPIC\t-\t", mqtt_topic if len(mqtt_topic) > 0 else "None")
    if os.environ.get('RTSP_TRANSPORT') is not None:
        print("RTSP TRANSPORT\t-\t", rtsp_transport if len(rtsp_transport) > 0 else "None")
    stream_started = False
    imgsz = 640
    dataset = LoadStreams(source, img_size=imgsz)
    print("[INFO] loading face detector model...")
    # load our serialized face detector model from disk
    prototxtPath = "deploy.prototxt.txt"
    weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=source).start()

    # Set up MQTT if required
    mqtt_client = None
    if len(mqtt_address) > 0:
        mqtt_img_out = io.BytesIO()
        if ':' in mqtt_address:
            host, port = mqtt_address.split(':')
            port = int(port)
        else:
            host, port = mqtt_address, 1883
        mqtt_client_id = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(10)])
        mqtt_client = mqtt.Client(f"facemask_detection_{mqtt_client_id}")
        mqtt_client.connect(host, port=port)
        mqtt_last_message = [0, 0]
        mqtt_last_message_time = time()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=1200, height=1200)

        mqtt_new_message = [0, 0]

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            pred_lgt = (1, 0) if mask > withoutMask else (0, 1)

            if mqtt_client is not None:
                # mqtt message format = (no_mask, mask)
                mqtt_new_message[1] += pred_lgt[0]
                mqtt_new_message[0] += pred_lgt[1]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "mask" if mask > withoutMask else "no_mask"
            color = (0, 255, 0) if label == "mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # print('Percentage CPU: ', psutil.cpu_percent())
            # print('Physical memory usage: ', psutil.virtual_memory())
            # print('memory % used:', psutil.virtual_memory()[2])


            # Stream results
        if len(output) > 0:

            # Output stream is images piped to ffmpeg
            if not stream_started:
                h, w = frame.shape[:-1]
                proccc = Popen([
                'ffmpeg',
                '-f', 'mjpeg',
                '-i', '-',
                '-c:v', 'copy',
                '-s', f'{w}x{h}',
                '-f', 'rtsp',
                '-rtsp_transport', rtsp_transport,
                output], stdin=PIPE)
                stream_started = True
                print(f"\n *** Launched RTSP Streaming at {output} ***\n\n")

            # Pipe image
            im0i = Image.fromarray(frame[..., ::-1], 'RGB')  # bgr in opencv to rgb
            im0i.save(proccc.stdin, 'JPEG')

        # Publish detections to MQTT
        if len(mqtt_address) > 0:
            # send update every 30s if nothing's happened
            if mqtt_new_message != mqtt_last_message or time() - mqtt_last_message_time > 30:
                mqtt_last_message = mqtt_new_message
                mqtt_last_message_time = time()

                # Build JSON message
                mqtt_json = {}
                mqtt_json['classes'] = []
                for n, cls_name in zip(mqtt_new_message, ['no_mask', 'mask']):
                    mqtt_json['classes'].append({'name': cls_name, 'count': n})
                mqtt_json["input_rtsp_url"] = source

                # Encode image to send over MQTT
                img = Image.fromarray(frame)
                img.save(mqtt_img_out, format="png")
                img_b64 = base64.b64encode(mqtt_img_out.getvalue())
                img_b64_str = img_b64.decode('ascii')
                mqtt_json["snapshot"] = img_b64_str

                # Send MQTT
                mqtt_client.publish(mqtt_topic, json.dumps(mqtt_json))

                # Clear BytesIO
                mqtt_img_out.seek(0)
                mqtt_img_out.truncate()

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        #
        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break

    # do a bit of cleanup
    # cv2.destroyAllWindows()
    # vs.stop()


if __name__ == '__main__':
    
    # INPUT_RTSP_URL
    source = ""
    if os.environ.get('INPUT_RTSP_URL') is not None:
        source = os.environ['INPUT_RTSP_URL']
    if len(source) == 0:
        raise Exception("Input RTSP URL must be specified ($INPUT_RTSP_URL)")
    if not (source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))):
        raise Exception("The specified input is not a video stream.")
    if source.isnumeric():
        source = int(source)

    # OUTPUT_RTSP_URL
    output = ""
    if os.environ.get('OUTPUT_RTSP_URL') is not None:
        output = os.environ['OUTPUT_RTSP_URL']
    
    # MQTT_URL & MQTT_TOPIC
    mqtt_address = ""
    mqtt_topic = ""
    if os.environ.get('MQTT_URL') is not None:
        mqtt_address = os.environ['MQTT_URL']
    if os.environ.get('MQTT_TOPIC') is not None:
        mqtt_topic = os.environ['MQTT_TOPIC']
    if len(mqtt_address) > 0 and len(mqtt_topic) == 0:
        raise Exception("MQTT topic must be specified if address is given ($MQTT_TOPIC)")
    if len(mqtt_topic) > 0 and len(mqtt_address) == 0:
        raise Exception("MQTT broker address must be specified if topic is specified ($MQTT_URL). Format = host or host:port (default port is 1883)")
    
    # RTSP_TRANSPORT
    rtsp_transport = 'tcp'
    if os.environ.get('RTSP_TRANSPORT') is not None:
        rtsp_transport = os.environ['RTSP_TRANSPORT']
    if rtsp_transport not in ['tcp', 'udp']:
        raise Exception("RTSP output stream transport layer protocol must be tcp or udp")
    
    detect(source, output, mqtt_address, mqtt_topic, rtsp_transport)
