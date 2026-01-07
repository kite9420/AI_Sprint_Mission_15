import streamlit as st
import pandas as pd
import numpy as np
import torch
import model
from model import ImageClassifier
import emoji

@st.cache_resource
def load_classifier():
    return ImageClassifier()

classifier = load_classifier()

st.set_page_config(page_title="Image_classification_", layout="wide")

st.title("Image Classification Application")
st.write("이미지를 분류하기 위해 분류할 이미지를 업로드하세요.")



# 1. CSS 주입: 모든 이미지를 동일한 높이로 고정
st.markdown("""
    <style>
    /* 이미지 컨테이너 높이 고정 및 정렬 */
    div[data-testid="stImage"] img {
        height: 400px;       /* 원하는 높이로 조절하세요 */
        object-fit: cover;   /* 비율을 유지하면서 고정된 크기에 꽉 채움 */
        width: 100%;         /* 너비는 컬럼에 맞춤 */
        border-radius: 10px; /* 모서리 둥글게 */
    }
    </style>
    """, unsafe_allow_html=True)


def get_emoji_prefix(label):
    clean_label = label.lower().replace(" ", "_")
    alias = f":{clean_label}:"
    converted = emoji.emojize(alias, language='alias')
    if converted == alias:
        return ""
    return converted + " "

# 2. 업로드된 이미지 표시
st.sidebar.header("Upload image")
uploaded_files = st.sidebar.file_uploader(
    "이미지를 선택하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    num_images = len(uploaded_files)
    classify_btn = st.sidebar.button("이미지 분류 시작하기", use_container_width=True)
    st.title("이미지 미리보기")
    st.write("---")
    
    # 한 줄에 보여줄 컬럼 개수 (예: 4개)
    cols_per_row = 4
    cols = st.columns(cols_per_row)

for idx, file in enumerate(uploaded_files):
        with cols[idx % cols_per_row]:
            with st.container(border=True): 
                st.image(file, caption=f"Image: {file.name}", use_container_width=True)
                
                result_area = st.empty() 

                if classify_btn:
                    with st.spinner("분석중..."):
                        predictions = classifier.predict(file, top_k=5)
                    
                    top_prediction = predictions[0]
                    label_name = top_prediction['label']
                    score = top_prediction['score'] 
                    prefix = get_emoji_prefix(label_name)
                    
                    # #### 3. 결과 표시 ####
                    st.success(f"**{prefix}{label_name} ({score*100:.2f}%)**")
                    
                    chart_df = pd.DataFrame(predictions)
                    st.bar_chart(data=chart_df, x='label', y='score', use_container_width=True)
                else:
                    st.info("분류하려면 '이미지 분류 시작하기' 버튼을 클릭하세요.")