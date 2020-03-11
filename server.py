# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 02:04:25 2020

@author: ACER
"""

import socket
import pickle
import struct
import numpy as np
import cv2

#yolo train
x_obj = 0
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
    
req = 'bottle'
for j in range(len(classes)):
        if (str(req) == str(classes[j])):
            idx = j
            print("idx=",idx)
            break
        else:
            idx = -1



host = ""
port = 9999
s=socket.socket()
print('Socket created')

s.bind((host,port))
print('Socket bind complete')
s.listen(5)
print('Socket now listening')

conn,addr=s.accept()
print("Connection with: IP-"+str(addr[0])+" Port-"+str(addr[1]))
s.setblocking(1)

data = b""
rec_size = struct.calcsize(">L")
while True:
    while(len(data)<rec_size):
        data+=conn.recv(4096)
    print("Started receiving data")
    inp_msg_size = data[:rec_size]
    data = data[rec_size:]
    msg_size = struct.unpack(">L",inp_msg_size)[0]
    while(len(data)<msg_size):
        data+= conn.recv(4096)
    fr_data = data[:msg_size]
    data = data[msg_size:]
    
    frame = pickle.loads(fr_data,fix_imports = True,encoding = "bytes")
    frame = cv2.imdecode(frame,cv2.IMREAD_COLOR)
    
    height,width,channels = frame.shape
    layers = net.getLayerNames()
    outputs = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame , 0.00392, (416,416),(0,0,0),True,crop = False)
    net.setInput(blob)
    out = net.forward(outputs)
    boxes = []
    class_ids = []
    confidences = []
    for i in out:
        for detect in i:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if (confidence>0.5):
                c_x = int(detect[0]*width)
                c_y = int(detect[1]*height)
                w = int(detect[2]*width)
                h = int(detect[3]*height)
                x = int(c_x - (w/2))
                y = int(c_y - (h/2))
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                
                boxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(confidence)
    
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)           
    n = len(boxes)
    font = cv2.FONT_HERSHEY_PLAIN
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_green = np.array([33,80,40]) 
    upper_green = np.array([102,255,255])
    green_mask = cv2.inRange(hsv,lower_green,upper_green)  
    
    contours_g,_ = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for contour in contours_g:
        area = cv2.contourArea(contour)
        if area>500:
            print("hello")
            cv2.drawContours(frame,contour,-1,(0,255,0),3)
            bounding_box = cv2.boundingRect(contour)
            x_band = bounding_box[0]+(bounding_box[2]/2)
            print('x_band =',x_band)
            cv2.rectangle(frame,(bounding_box[0],bounding_box[1]),(bounding_box[0]+bounding_box[2],bounding_box[1]+bounding_box[3]),3)

            for i in range(n):
                name = str(classes[class_ids[i]])
                if class_ids[i]==idx:
                    print(name)
                    x,y,w,h = boxes[i]
                    x_obj = x + (w/2)
                    print(x_obj)
                    y_obj = y + (h/2)
                    w_obj = w 
                    h_obj = h
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,name,(int(x+w/2),y-5),font,1,(0,0,0),2)
                    d = x_band - x_obj
                    distance = abs(d)
                    print('distance =',distance)
                    if(distance<3*w):
                        freq = 255 - distance
                        print("freq = ",freq)
                    else:
                        freq = 0
                        print("freq =",freq)
                    #conn.send(str.encode(freq))
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
s.close()
cv2.destroyAllWindows()