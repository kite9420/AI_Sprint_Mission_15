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

st.set_page_config(page_title="Image Classification", layout="wide")
st.title("Image Classification Application")

# 1. 이미지 높이 고정 CSS
st.markdown("""
    <style>
    div[data-testid="stImage"] img {
        height: 300px;
        object-fit: cover;
        width: 100%;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_emoji_prefix(label):
    words = label.lower().replace(",", " ").replace("_", " ").split()
    
    for word in reversed(words):
        alias = f":{word}:"
        converted = emoji.emojize(alias, language='alias')
        
        if converted != alias:
            return converted + " "
            
    return ""

# 2. 사이드바 설정
st.sidebar.header("Upload image")
uploaded_files = st.sidebar.file_uploader(
    "이미지를 선택하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)


if uploaded_files:
    classify_btn = st.sidebar.button("이미지 분류 시작하기", width='stretch')
    
    st.write("---")
    cols_per_row = 4
    cols = st.columns(cols_per_row)

    for idx, file in enumerate(uploaded_files):
        with cols[idx % cols_per_row]:
            with st.container(border=True): # 테두리 박스 추가
                st.image(file, caption=f"File: {file.name}", width='stretch')
                
                if classify_btn:
                    with st.spinner("분석 중..."):
                        predictions = classifier.predict(file, top_k=5)
                    
                    # 데이터 추출
                    top_pred = predictions[0]
                    label_name = top_pred['label']
                    score = top_pred['score'] # <-- score 변수 확실히 정의
                    prefix = get_emoji_prefix(label_name)
                    
                    # 결과 출력
                    st.success(f"**{prefix}{label_name} ({score*100:.2f}%)**")
                    
                    # 차트 출력
                    chart_df = pd.DataFrame(predictions)
                    st.bar_chart(data=chart_df, x='label', y='score', width='stretch')
                else:
                    st.info("분류 버튼을 눌러주세요.")
else:
    st.info("사이드바에서 이미지를 업로드하세요.")