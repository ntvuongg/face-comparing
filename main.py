import os
import mtcnn
import VGG16
import streamlit as st
from PIL import Image
from scipy.spatial import distance
import numpy as np
from pathlib import Path

# COSINE_THRESHOLD = 0.75
# --------------------------------
# """ Config""""
hide_menu = """
<style>
footer{
    visibility: visible;
}
footer:after{
    content:' Developed by @Nguyen Thanh Vuong';
    display: block;
    position: relative;
    color: black;
}
</style>
"""

st.set_page_config(page_title='Demo Project', page_icon="🖖")
# --------------------------------
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

INPUT_DIR = os.listdir('upload_img')

def upload(index):
    UPLOAD_FOLDER = 'upload_img'
    st.markdown(f"**Person {index}**")
    p_upload = st.selectbox("Select a way to upload your image", ["Upload from my computer", "Take a pictute"], key=index)
    if p_upload == "Upload from my computer":
        img = st.file_uploader("Choose an image", type=["png","jpg","jpeg"], key=index)
        if img is not None:
            st.image(Image.open(img))
            with open(os.path.join(UPLOAD_FOLDER, f"{index}.jpg"),"wb") as f:
                f.write((img).getbuffer())
            st.success('Image uploaded successfully!')
            return True

    else:
        st.title("Webcam Live")
        st.info("Please allowing camera to access on this website first!")
        # img = webCam(index)
        img = st.camera_input("Please take a photo with a person in the center of the image!", key=index)
        if img is not None:
            with open(os.path.join(UPLOAD_FOLDER, f"{index}.jpg"),"wb") as f:
                f.write((img).getbuffer())
            st.success("Take image successfully!")
            return True
    return False

# """Footer"""
st.markdown(hide_menu, unsafe_allow_html=True)
# """ Side bar"""
st.sidebar.title("Navigation Bar")
st.sidebar.markdown("----")
SIDEBAR_OPTIONS = [ "Project info", "Application", "Contact"]
app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

if app_mode == "Project info":
    intro_markdown = read_markdown_file("Project_info.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
    expander_faq = st.expander("More About My Project")
    expander_faq.write("Hi there! If you have any questions about my project, or simply want to check out the source code, please visit my github repo: https://github.com/ntvuongg")


elif app_mode == "Application":
    # """  Main UI """
    cmp_method = st.sidebar.selectbox("Select a method to evaluate", ["You select...", "Cosine similarity", "Euclidean distance (L2 Norm)"])
    st.title("Simple Face Comparing App")
    st.markdown("----")
    check_1 = upload(1)
    st.markdown("----")
    check_2 = upload(2)

    col1, col2, col3 = st.columns([1.5,1,1])
    cmp_btn = col2.button("Compare")

    if cmp_btn:
        if cmp_method == "You select...":
            st.warning("Please select evaluation method!")
        elif not(check_1) or not(check_2):
            st.warning("Missing image! Please check again!")
        else:
            face_1 = mtcnn.getFace('upload_img/1.jpg')
            face_1.save('upload_img/1.jpg')
            face_2 = mtcnn.getFace('upload_img/2.jpg')
            face_2.save('upload_img/2.jpg')
            model = VGG16.init_model()
            vector_p1 = VGG16.extract_vector(model, 'upload_img/1.jpg')
            vector_p2 = VGG16.extract_vector(model, 'upload_img/2.jpg')
            
            if cmp_method == "Cosine similarity":
                alpha =  1 - distance.cosine(vector_p1, vector_p2)
                st.info(f'Similarity: {round(alpha*100, 2)}%')

            elif cmp_method == "Euclidean distance (L2 Norm)":
                dist = np.linalg.norm(vector_p1 - vector_p2)
                st.info(f'Distance: {dist}')
            
        cmp_btn = False

elif app_mode == "Contact":
    about_markdown = read_markdown_file("about.md")
    st.markdown(about_markdown, unsafe_allow_html=True)
    expander_faq = st.expander("More About My Project")
    expander_faq.write("Hi there! If you have any questions about my project, or simply want to check out the source code, please visit my github repo: https://github.com/ntvuongg")
    

