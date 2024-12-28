from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from models.main import get_features_extractor, get_age_classfiier, get_face_detector
import torch
import torch.nn.functional as F
import numpy as np

api = Flask(__name__)

@api.get('/')
def root():
    return jsonify({'message':'welcome to the api'}), 200

@api.post('/features')
def get_features():
    try:
        feature_extractor:torch.nn.Sequential = get_features_extractor()
        feature_extractor.eval()

        if 'inputs' not in request.json:
            raise Exception('There is no inputs')
        
        if not isinstance(request.json['inputs'],list):
            raise Exception('Wrong input object type')
        
        inputs = request.json['inputs']
        inputs_tensor = torch.tensor(inputs,dtype=torch.float32)

        if inputs_tensor.ndim != 4 or inputs_tensor.size(1) != 3:
            raise Exception('Wrong input shape')
        
        with torch.no_grad():
            features:torch.Tensor = feature_extractor(inputs_tensor)

        return jsonify({'predictions':features.tolist()}), 200
    except Exception as e:
        return jsonify({'error':str(e)}), 400
    
@api.post('/face_detection')
def face_detection():
    try:
        clf:LogisticRegression = get_face_detector()

        if 'inputs' not in request.json:
            raise Exception('There is no inputs')
        
        if not isinstance(request.json['inputs'],list):
            raise Exception('Wrong input object type')
        
        inputs = request.json['inputs']
        inputs_array = np.array(inputs)

        if inputs_array.ndim != 2 or inputs_array.shape[1] != 960:
            raise Exception('Wrong input shape')
        
        probability = clf.predict_proba(inputs_array)[:,1].item()

        return jsonify({'predictions':probability}),200
    
    except Exception as e:
        return jsonify({'error':str(e)}), 400

@api.post('/classify_age')
def classify_age():
    try:
        age_classifier = get_age_classfiier()
        age_classifier.eval()

        if 'inputs' not in request.json:
            raise Exception('There is no inputs')
        
        if not isinstance(request.json['inputs'], list):
            raise Exception('Wrong input object type')
        
        inputs = request.json['inputs']
        inputs_tensor = torch.tensor(inputs,dtype=torch.float32)

        if inputs_tensor.ndim != 2 or inputs_tensor.size(1) != 960:
            raise Exception('Wrong input shape')
        
        with torch.no_grad():
            pred = age_classifier(inputs_tensor)
            probability = F.softmax(pred,dim=1)
            probability = torch.round(probability,decimals=4)

        output = probability.squeeze(0).tolist()

        return jsonify({'predictions':output}),200
    
    except Exception as e:
        return jsonify({'error':str(e)}), 400

