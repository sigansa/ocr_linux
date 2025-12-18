#!/usr/bin/env python3
"""
Qwen2-VL 초소형 파인튜닝 (10개 샘플만)
실제로 완료시켜서 전후 비교
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from PIL import Image
import json
from tqdm import tqdm
import os

print("="*60)
print("🎯 초소형 파인튜닝 (10개 샘플, 실제 완료 목표)")
print("="*60)

# 4비트 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("\n📦 모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# LoRA 적용
print("🔧 LoRA 적용 중...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

lora_config = LoraConfig(
    r=8,  # 작게
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 핵심만
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 데이터 로드
print("\n📊 데이터 로딩 중...")
with open("/root/data/deepseek_ocr/train_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

# 딱 10개만
train_samples = train_data[:10]
print(f"학습 샘플: {len(train_samples)}개")
print(f"예상 시간: ~2-3분\n")

# 옵티마이저
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

# 학습 (1 epoch만)
print("🚀 학습 시작!")
print("-"*60)

model.train()
total_loss = 0
success_count = 0

for idx, item in enumerate(tqdm(train_samples, desc="학습 진행")):
    try:
        messages = item["messages"]
        
        # 데이터 추출
        image_path = messages[0]["content"][0]["image"]
        user_text = messages[0]["content"][1]["text"]
        assistant_text = messages[1]["content"][0]["text"]
        
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        
        # 대화 구성
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        
        # 프로세싱
        text_prompt = processor.apply_chat_template(conversation, tokenize=False)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Forward & Backward
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        
        # Update (매 샘플마다)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        success_count += 1
        
    except Exception as e:
        print(f"\n⚠️ 샘플 {idx} 실패: {e}")
        continue

avg_loss = total_loss / max(success_count, 1)

print(f"\n✅ 학습 완료!")
print(f"   - 처리 샘플: {success_count}/{len(train_samples)}")
print(f"   - 평균 Loss: {avg_loss:.4f}")

# 모델 저장
print("\n💾 모델 저장 중...")
save_dir = "/root/data/qwen2vl_tiny_finetuned"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

print(f"✅ 저장 완료: {save_dir}")
print("\n" + "="*60)
print("🎉 파인튜닝 완료! 이제 비교 평가를 진행하세요.")
print("="*60)
