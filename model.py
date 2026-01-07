import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

MODEL_NAME = "google/vit-base-patch16-224"
class ImageClassifier:
    def __init__(self):
        # 1. 프로세서와 모델 로드 (최초 1회 실행)
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    def predict(self, image_file, top_k=5):
        """
        이미지 파일을 받아 label과 score를 반환합니다.
        """
        # Streamlit의 UploadedFile을 PIL Image로 변환
        image = Image.open(image_file).convert("RGB")
        
        # 2. 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")

        # 3. 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # 4. 상위 k개의 확률과 인덱스 추출
        top_probs, top_indices = torch.topk(probs, top_k)

        # 5. 결과 리스트 생성
        results = []
        for i in range(top_k):
            label = self.model.config.id2label[top_indices[0][i].item()]
            score = top_probs[0][i].item()
            results.append({"label": label, "score": score})


        return results