#!/usr/bin/env python3
"""
Qwen2-VL 간단한 추론 테스트
파인튜닝 전에 모델 작동 확인
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

print("📦 모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("✅ 모델 로딩 완료\n")

# 테스트 이미지 (첫 번째 학습 샘플)
test_image_path = "/root/data/deepseek_ocr/filtered_train/images/cat1_1.jpg"
image = Image.open(test_image_path).convert("RGB")

# 대화 구성
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "이 간판 이미지의 텍스트를 읽어주세요."}
        ]
    }
]

# 텍스트 생성
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
).to("cuda")

print("🔍 간판 텍스트 인식 중...")
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)

generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# 응답 추출 (assistant: 이후 텍스트)
if "assistant\n" in generated_text:
    answer = generated_text.split("assistant\n")[-1].strip()
else:
    answer = generated_text

print(f"✅ 인식 결과: {answer}")
print(f"\n전체 출력:\n{generated_text}")
