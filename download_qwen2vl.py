#!/usr/bin/env python3
"""
Qwen2-VL 모델 다운로드 스크립트
모든 파일을 미리 다운로드
"""

from transformers import AutoProcessor, AutoModel

print("📥 Qwen2-VL 모델 다운로드 시작...")
model_id = "Qwen/Qwen2-VL-2B-Instruct"

print("1/2 Processor 다운로드...")
processor = AutoProcessor.from_pretrained(model_id)

print("2/2 Model 다운로드...")
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
)

print("✅ 다운로드 완료!")
print(f"캐시 위치: ~/.cache/huggingface/hub/")
