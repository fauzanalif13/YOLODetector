import cv2
import numpy as np
import time
#YOLO detection - realtime object detector

#fps logic
prev_frame_time = 0
new_frame_time = 0

confidenceThresHold = 0.6
nmsThreshold = 0.3 #semakin kecil nilai ini, semakin besar terjadi pengurangan
timer = cv2.getTickCount()
#fps = int(cv2.getTickFrequency() / (cv2.getTickCount()-timer))

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

pathClass = 'coco.names'
classNames = []
with open(pathClass,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

#yolov3_320 nya ta' ganti menjadi dri github, bkn dri yolo website
#320, 480, 640, utk mendeteksi lebih akurat
#tiny utk mendeteksi lebih cepat
modelConf = 'yolov3_3202.cfg'
modelWeights = 'yolov3_320.weights'

net = cv2.dnn.readNetFromDarknet(modelConf,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#fungsi utk mendeteksi object
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confValue= []

    for output in outputs:
        for det in output:
            #kita ingin membiarkan 5 value pertama (cx,cy,w,h, confidence)
            scores = det[5:]
            #kita ingin mendeteksi probability mana yg paling tinggi
            classId = np.argmax(scores)
            #kemudian kita ingin menyimpannya di confidence
            confidence = scores[classId]
            if confidence > confidenceThresHold:
                #menentukan lokkasi w dan h
                w, h = int(det[2]*wT), int(det[3]*hT)
                #menentukan lokasi x dan y
                x,y =  int((det[0]*wT) -w/2), int((det[1]*hT) -h/2)
                #nilai x,y,w,h ditambahkan ke bbox
                bbox.append([x,y,w,h])
                #nilai classId ditambahkan ke classIds
                classIds.append(classId)
                #nilai confidence (float) ditambahkan ke confValue
                confValue.append(float(confidence))
    #kita ingin melihat berapa object yg dideteksi
    #print (len(bbox))

    #kita ingin menghindari overlapping box
    #indices adalah indeks
    indices = cv2.dnn.NMSBoxes(bbox, confValue, confidenceThresHold, nmsThreshold)
    for i in indices:
        i = i[0] #kita ingin mengeluarkan bracket deteksi
        box = bbox[i] #lokasi bounding box
        x,y,w,h = box[0], box[1], box [2], box[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confValue[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)

while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (lebar,lebar), [0,0,0],1, crop=False)
    #print (blob)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    #net.getUnconnectedOutLayers()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    #print (net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    #disini ada 3 karena dlm yolo, itu memproses 3 layer
    #selengkapnya bisa dilihat pada yoloWork.png
    # print (outputs[0].shape)
    # print (outputs[1].shape)
    # print (outputs[2].shape)
    #ini akan menghasilkan (x,y) = (1200,85) (4800,85) (19200,85)
    #(x,y) = x adalah v value, tingkat pendeteksian
    # y adalah prediksi kemungkinan pada setiap kelas
    findObjects(outputs, img)

    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(img, "FPS live:{0}".format(fps), (7, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

    cv2.imshow("YOLO detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()