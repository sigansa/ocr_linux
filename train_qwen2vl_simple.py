#!/usr/bin/env python3
"""
Qwen2-VL 파인튜닝 스크립트 (간소화 버전)
데이터를 미리 처리하고 저장하는 방식
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import os

print("📦 설정 시작...")

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드
print("🔄 모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 모델 준비
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# LoRA 설정 (더 보수적인 설정)
print("🔧 LoRA 적용 중...")
lora_config = LoraConfig(
    r=8,  # rank 낮춤
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 핵심 모듈만 타겟
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# LoRA 파라미터가 학습 가능한지 확인
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
        print(f"✓ {name}: requires_grad={param.requires_grad}")

# 데이터셋 로드
print("\n📊 데이터셋 로딩 중...")
with open("/root/data/deepseek_ocr/train_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

with open("/root/data/deepseek_ocr/val_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    val_data = [json.loads(line) for line in f]

# 샘플만 사용 (빠른 테스트)
train_samples = train_data[:50]  # 50개만
val_samples = val_data[:10]  # 10개만

print(f"- 학습 샘플: {len(train_samples)}개")
print(f"- 검증 샘플: {len(val_samples)}개")

# 학습 함수
def train_step(batch_data, model, optimizer):
    """단일 학습 스텝"""
    model.train()
    total_loss = 0
    
    for item in batch_data:
        messages = item["messages"]
        
        # 이미지와 텍스트 추출
        image_path = messages[0]["content"][0]["image"]
        user_text = messages[0]["content"][1]["text"]
        assistant_text = messages[1]["content"][0]["text"]
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}")
            continue
        
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
        
        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        total_loss += loss.item()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss / len(batch_data)

# 평가 함수
def evaluate(val_data, model):
    """모델 평가"""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for item in val_data:
            messages = item["messages"]
            
            # 이미지와 텍스트 추출
            image_path = messages[0]["content"][0]["image"]
            user_text = messages[0]["content"][1]["text"]
            assistant_text = messages[1]["content"][0]["text"]
            
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                continue
            
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
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1
    
    return total_loss / max(count, 1)

# 옵티마이저 설정
print("\n🔧 옵티마이저 설정...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 학습 시작
print("\n🚀 학습 시작!")
num_epochs = 3
batch_size = 4

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*50}")
    
    # 학습
    epoch_loss = 0
    num_batches = (len(train_samples) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(train_samples), batch_size), desc="Training"):
        batch = train_samples[i:i+batch_size]
        loss = train_step(batch, model, optimizer)
        epoch_loss += loss
    
    avg_train_loss = epoch_loss / num_batches
    
    # 평가
    print("\n📊 검증 중...")
    val_loss = evaluate(val_samples, model)
    
    print(f"✅ Epoch {epoch+1} 완료")
    print(f"   - 학습 Loss: {avg_train_loss:.4f}")
    print(f"   - 검증 Loss: {val_loss:.4f}")
    
    # 체크포인트 저장
    if (epoch + 1) % 1 == 0:
        save_path = f"/root/data/qwen2vl_finetuned/checkpoint-epoch-{epoch+1}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"💾 체크포인트 저장: {save_path}")

# 최종 모델 저장
print("\n💾 최종 모델 저장 중...")
final_path = "/root/data/qwen2vl_finetuned/final"
os.makedirs(final_path, exist_ok=True)
model.save_pretrained(final_path)
processor.save_pretrained(final_path)

print(f"\n✅ 학습 완료!")
print(f"📁 저장 위치: {final_path}")
