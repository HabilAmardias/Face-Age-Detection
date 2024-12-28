from torchvision import models
import torch
import pickle

def get_features_extractor():
    model = models.mobilenet_v3_large()
    model.classifier[3] = torch.nn.Linear(1280, 4)

    model.load_state_dict(
        torch.load(
            'models/age_detector.pth',
            map_location='cpu',
            weights_only=True
        )
    )

    feature_extractor = torch.nn.Sequential()
    feature_extractor.add_module('features',model.features)
    feature_extractor.add_module('avgpool',model.avgpool)

    return feature_extractor

def get_age_classfiier():
    model = models.mobilenet_v3_large()
    model.classifier[3] = torch.nn.Linear(1280, 4)

    model.load_state_dict(
        torch.load(
            'models/age_detector.pth',
            map_location='cpu',
            weights_only=True
        )
    )

    return model.classifier

def get_face_detector():
    with open('models/classifier.pkl','rb') as f:
        clf = pickle.load(f)
    return clf