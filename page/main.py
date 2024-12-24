import streamlit as st
import requests
import os
from page.utils.encoder import get_class

st.set_page_config('Age Detection',layout='centered')
st.title('Face Age Detector')

api_url = os.environ['API_URL']
class_list = get_class()

uploaded = None
camera = None

if "disable_camera" not in st.session_state:
    st.session_state.disable_camera = False
if "disable_upload" not in st.session_state:
    st.session_state.disable_upload = False

def toggle_camera():
    st.session_state.disable_camera = not st.session_state.disable_camera
    st.session_state.disable_upload = False

def toggle_upload():
    st.session_state.disable_upload = not st.session_state.disable_upload
    st.session_state.disable_camera = False

st.markdown("""
This project implement deep learning model (MobileNetV3 from [PyTorch](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html#torchvision.models.mobilenet_v3_large)) 
for Face Age Detection, fine-tuned on [Face Dataset Here](https://susanqq.github.io/UTKFace/). The implemented model here is still experimental. If your uploaded image is rejected then either

- There is no human face in the image, or
- Poor lighting condition or the face is not close enough, or
- My App's Fault
""")

st.write("The image you have uploaded will not be saved by the system.")

if not st.session_state.disable_upload:
    uploaded = st.file_uploader(
        label='Upload an Image Here',
        accept_multiple_files=False,
        type=['png','jpg','jpeg','bmp'],
        on_change=toggle_camera
    )

if not st.session_state.disable_camera:
    camera = st.camera_input(
        "Take a Picture",on_change=toggle_upload
    )

if (uploaded and not camera) or (not uploaded and camera):
    if uploaded:
        files = {
            'file':(uploaded.name, uploaded, uploaded.type)
        }
    else:
        files = {
            'file':("camera_image.jpeg", camera, "image/jpeg")
        }
    try:
        response = requests.post(api_url,files=files)

        if response.status_code == 200:
            probability = response.json()['prediction']

            data = {}
            for i,class_ in enumerate(class_list):
                data[f'Age {class_} probability'] = [probability[i]]
            
            st.dataframe(data,
                         use_container_width=True)
        else:
            st.error(
                f'Failed to upload file: {response.json()['error']}'
            )
    except Exception as e:
        st.error(f"Error: {str(e)}")
            