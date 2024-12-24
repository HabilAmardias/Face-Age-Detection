from torchvision import models
import torch
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pickle
import numpy as np

def get_model():
    model = models.mobilenet_v3_large()
    model.classifier[3] = torch.nn.Linear(1280, 4)

    model.load_state_dict(
        torch.load(
            'models/age_detector.pth',
            map_location='cpu',
            weights_only=True
        )
    )
    return model

def get_classifier():
    with open('models/classifier.pkl','rb') as f:
        clf = pickle.load(f)
    return clf

if __name__ == '__main__':
    with mlflow.start_run() as run:
        model = get_model()

        feature_extractor = torch.nn.Sequential()
        feature_extractor.add_module('features',model.features)
        feature_extractor.add_module('avgpool',model.avgpool)

        model.eval()
        feature_extractor.eval()

        clf = get_classifier()
        
        mlflow.pytorch.log_model(feature_extractor,'age_detector_features',
                                 input_example=torch.randn((1,3,224,224),dtype=torch.float32).numpy())
        mlflow.pytorch.log_model(model.classifier,'age_detector_classifier',
                                 input_example=torch.randn((1,960),dtype=torch.float32).numpy())
        mlflow.sklearn.log_model(clf,'face_detector',
                                 input_example=np.random.normal(
                                     0,1,(1,960)
                                 ))
        print(run.info.run_id)
