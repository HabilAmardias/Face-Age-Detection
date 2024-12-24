import PIL.Image
from prefect import flow, task
import torch.nn.functional as F
from torchvision.transforms import v2
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import os
import torch
import PIL
from sklearn.linear_model import LogisticRegression


@task
def load_model():
    run_id = os.environ['RUN_ID']

    features_uri = f"runs:/{run_id}/age_detector_features"
    features_extractor = mlflow.pytorch.load_model(features_uri)

    classifier_uri = f"runs:/{run_id}/age_detector_classifier"
    age_classifier = mlflow.pytorch.load_model(classifier_uri)
    
    return features_extractor, age_classifier

@task
def load_classifier() -> LogisticRegression:
    run_id = os.environ['RUN_ID']
    clf_uri = f"runs:/{run_id}/face_detector"
    clf = mlflow.sklearn.load_model(clf_uri)
    return clf

@task
def get_transform() -> v2.Compose:
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean = [0.485, 0.456, 0.406], 
                     std = [0.229, 0.224, 0.225])
    ])
    return transform

@task
def extracting_features(upload, transform:v2.Compose,
                        features_extractor) -> torch.Tensor:
    features_extractor.eval()
    image = PIL.Image.open(upload).convert('RGB')
    image_tensor:torch.Tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features:torch.Tensor = features_extractor(image_tensor).squeeze((2,3))
    return features

@task
def get_score(clf:LogisticRegression,
            features:torch.Tensor) -> float:
    features_numpy = features.cpu().numpy()
    probs = clf.predict_proba(features_numpy)[:,1].item()

    return probs

@task
def model_predict(age_classifier, 
                  score:float,
                  features:torch.Tensor, 
                  threshold:float) -> list | None:
    age_classifier.eval()
    if score >= threshold:
        with torch.no_grad():
            pred = age_classifier(features)
            probability = F.softmax(pred,dim=1)
            probability = torch.round(probability,decimals=4)
        return probability.squeeze(0).tolist()
    else:
        return None

@flow
def predict(upload,
            threshold:float=0.5) -> list | None:
    transform = get_transform()
    features_extractor, age_classifier = load_model()
    clf = load_classifier()

    features = extracting_features(upload,transform,features_extractor)
    score = get_score(clf,features)
    
    return model_predict(age_classifier,score,features,threshold)
        


