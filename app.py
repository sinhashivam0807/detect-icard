import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, jsonify,request
import os
import json
import re
import base64
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

API_KEY=os.getenv("API_KEY")
def detectcard(image):
    model = load_model('keras_model.h5')
    labels = open('labels.txt', 'r').readlines()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    probabilities = model.predict(image)
    prediction=probabilities.flatten()
    if(prediction[0]>prediction[1]):
        output={"respone":"True"}
        return jsonify(output)
    else:
        output={"respone":"False"}
        return jsonify(output) 

app= Flask(__name__)
CORS(app)

@app.route('/detectcard', methods = ['POST'])
def ReturnJSON():
    try:
        if(request.headers.get('API_KEY')==API_KEY):
            img2get=(request.json.get('image'))
            with open("testimage.png", "wb") as fh:
                fh.write(base64.b64decode(img2get))
            img=cv2.imread('testimage.png')
            result=(detectcard(img))
            return result
        else:
            error={"error":"authorization error"}
            return jsonify(error)
    except:
        error={"error":"Bad request"}
        return jsonify(error)

if __name__ == '__main__':
    app.run(debug=True,ssl_context="adhoc")
