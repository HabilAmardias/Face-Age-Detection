import pickle

def get_class():
    with open('page/utils/encoder.pkl','rb') as f:
        encoder = pickle.load(f)
    return encoder