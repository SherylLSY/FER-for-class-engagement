#for getting engagement of class based on FE

import io, os
import cv2
import numpy as np
import pandas as pd
import argparse
import google.cloud
# Imports the Google Cloud client library
from google.cloud import vision
from decimal import Decimal
from numpy import random
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"StudentEngagement-ae34535b572f.json"

#Emotions
emo = ['Anger', 'Surprise', 'Sorrow', 'Joy']

likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
string = 'Neutral'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description='Process some image to find sentiment in faces (if any)')
ap.add_argument("-f", "--file_name", required=False, default="cut5r/1.jpg", help="path to image")
args = vars(ap.parse_args())

file_name = args["file_name"]

# Instantiates a client
vision_client = vision.ImageAnnotatorClient()

max_results=80

#open file
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image(content=content)

#receive result from Google Cloud Vision API
response = vision_client.face_detection(
    image=image, max_results=max_results)

#select only face emotion
faces = response.face_annotations

print('')
print('Number of faces: ', len(faces))
print('')

img = cv2.imread(file_name)

e = 'Engage'
d = 'Disengage'

for face in faces:
    Confidence_level = format(face.detection_confidence)

    face_vertices = ['({0},{1})'.format(vertex.x,vertex.y) for vertex in face.bounding_poly.vertices]

    x = face.bounding_poly.vertices[0].x
    y = face.bounding_poly.vertices[0].y
    x2 = face.bounding_poly.vertices[2].x
    y2 = face.bounding_poly.vertices[2].y

    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

    sentiment = [likelihood_name[face.anger_likelihood],
                likelihood_name[face.surprise_likelihood],
                likelihood_name[face.sorrow_likelihood],
                likelihood_name[face.joy_likelihood]]

    for item, item2 in zip(emo, sentiment):
        print(item, ": ", item2)

    if not(all(item == 'VERY_UNLIKELY' for item in sentiment) ):

        if any(item == 'VERY_LIKELY' for item in sentiment):
            state = sentiment.index('VERY_LIKELY')
            # the order of enum type Likelihood is:
            #'LIKELY', 'POSSIBLE', 'UNKNOWN', 'UNLIKELY', 'VERY_LIKELY', 'VERY_UNLIKELY'
            # it makes sense to do argmin if VERY_LIKELY is not present, one would espect that VERY_LIKELY
            # would be the first in the order, but that's not the case, so this special case must be added
        else:
            state = np.argmin(sentiment)

        string = emo[state]


        print("*Emotion -> " + string )

        if (string=='Joy'  ):
            print("*Engagement -> " + e); #e = engagement
            cv2.putText (img,e, (x,y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            #cv2.putText(img,'Engage', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        else:
            print("*Engagement -> " + d); #d = disengagement
            cv2.putText (img,d, (x,y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    else:
        print("*Emotion -> Neutral" )
        print("*Engagement -> " + e)
        cv2.putText (img,e, (x,y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    num = Decimal(Confidence_level)
    con_level_2dp = round(num, 3)

    #https://www.december.com/html/spec/colorrgbadec.html
    #https://www.html.am/html-codes/color/color-scheme.cfm?rgbColor=0,0,255
    #cv2.putText (img,string, (x,y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 2)

    #https://www.abbreviations.com/abbreviation/Confidence+Level
    #CL = Confidence Level
    cv2.putText (img, 'CL: ' + str(con_level_2dp), (x,y2-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    print('Detection Confidence : ' + Confidence_level)
    print('Face bound: {0}'.format(', ' .join(face_vertices)+'\n' ))

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.imwrite('cut5-output-eng/output_1.jpg',img)
