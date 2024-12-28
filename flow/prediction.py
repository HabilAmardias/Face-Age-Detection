import PIL.Image
from torchvision.transforms import v2
import torch
import PIL
import requests
import os

def get_transform() -> v2.Compose:
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean = [0.485, 0.456, 0.406], 
                     std = [0.229, 0.224, 0.225])
    ])
    return transform

def extracting_features(upload, transform:v2.Compose) -> torch.Tensor:
    model_api_url = os.environ['API_MODEL_URL']
    url = f'{model_api_url}/features'
    image = PIL.Image.open(upload).convert('RGB')
    image_tensor:torch.Tensor = transform(image).unsqueeze(0)
    data = {'inputs':image_tensor.tolist()}
    headers = {'Content-Type':'application/json'}
    try:
        response = requests.post(url=url,
                                 json=data,
                                 headers=headers)
        if response.status_code == 200:
            preds = response.json()['predictions']
            features = torch.tensor(preds,dtype=torch.float32)
            features = features.squeeze((2,3))
            return features
        else:
            raise Exception(response.json()['error'])
    except Exception as e:
        return str(e)

def get_score(features:torch.Tensor) -> float:
    model_api_url = os.environ['API_MODEL_URL']
    url = f'{model_api_url}/face_detection'
    data = {'inputs':features.tolist()}
    headers = {'Content-Type':'application/json'}
    try:
        response = requests.post(url=url,
                                 json=data,
                                 headers=headers)
        if response.status_code == 200:
            preds = response.json()['predictions']
            return preds
        else:
            raise Exception(response.json()['error'])
    except Exception as e:
        return str(e)

def model_predict(score:float,
                  features:torch.Tensor, 
                  threshold:float) -> list | None:
    model_api_url = os.environ['API_MODEL_URL']
    url = f'{model_api_url}/classify_age'
    data = {'inputs':features.tolist()}
    headers = {'Content-Type':'application/json'}
    if score < threshold:
        return None
    try:
        response = requests.post(url=url,
                                 json=data,
                                 headers=headers)
        if response.status_code == 200:
            preds = response.json()['predictions']
            return preds
        else:
            raise Exception(response.json()['error'])
    except Exception as e:
        return str(e)

def predict(upload,
            threshold:float=0.5) -> list | None:
    try:
        transform = get_transform()

        features = extracting_features(upload,transform)
        score = get_score(features)
        
        return model_predict(score,features,threshold)
    except Exception as e:
        return str(e)
        


