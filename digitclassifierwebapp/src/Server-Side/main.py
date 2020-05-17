# load Flask 
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import io, base64
import numpy as np
from keras import models

app = Flask(__name__)
CORS(app)

# define a process_image function as an endpoint 
@app.route("/process_image", methods=['GET','POST'])
def process_image():
    # access data that was passed 
    data_uri = request.data

    img = GetImage(data_uri)
    res = PredictImage(img)

    pred_value = np.argmax(res)
    accuracy = max(res)

    return jsonify({"Result": pred_value, "Accuracy": accuracy})

#Processes image to a format which can be passed to the model 
def GetImage(data):
    img = Image.open(io.BytesIO(base64.b64decode(data.split(',')[1])))

    img = img.resize((28, 28))

    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)

    #reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255

    return img
    
#Gets the results by passing in the image
def PredictImage(img):
    model = model.load_model("mnist.h5")
    result = model.predict([img])[0]
    return result
