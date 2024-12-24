from flask import Flask,request,jsonify
from flask_cors import cross_origin
from flow.prediction import predict
import os

api = Flask(__name__)

CLIENT_URL = os.environ['CLIENT_URL']
CONTENT_TYPE = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp']

@api.get('/')
def root():
    return jsonify({'message':'welcome to the api'}), 200

@api.post("/predict")
@cross_origin(origins=[CLIENT_URL])
def get_prediction():
    try:
        if 'file' not in request.files:
            raise Exception('No File Uploaded')
        
        file = request.files['file']
        if file.content_type not in CONTENT_TYPE:
            raise Exception('Wrong File Type')
        
        pred = predict(file)

        if not isinstance(pred,list):
            raise Exception('Image Rejected')
           
        return jsonify({"prediction":pred}), 200
            
    except Exception as e:
        return jsonify({"error":str(e)}), 400
