from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import math
import main

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            if(score < scoreThresh):
                continue
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))
    return [detections, confidences]

image = cv2.imread('testing.png')
orig = image.copy()
(H, W) = image.shape[:2]
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

net = cv2.dnn.readNet("frozen_east_text_detection.pb")
blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)

outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")

net.setInput(blob)
out = net.forward(outputLayers)

scores = out[0]
geometry = out[1]

boxes = []
confidences = []

[boxes, confidences] = decode(scores, geometry, 0.5)
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

n = len(boxes)
font = cv2.FONT_HERSHEY_PLAIN
height_ = image.shape[0]
width_ = image.shape[1]
rW = width_/ float(320)
rH = height_/ float(320)
for i in indices:
        vertices = cv2.boxPoints(boxes[i[0]])

        for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
        
        cv2.rectangle(image,(vertices[0][0],vertices[0][1]),(vertices[2][0],vertices[2][1]),(0,255,9),2)
        # main.init(image,vertices[0][0],vertices[0][1],vertices[2][0],vertices[2][1])

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()