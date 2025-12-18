#!/usr/bin/env python3
"""
Qwen2-VL 초간단 파인튜닝
수동 학습 루프로 최소한의 파인튜닝만 수행
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from PIL import Image
import json
from tqdm import tqdm
import os

print("📦 설정 시작...")

# 4비트 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
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

# LoRA 적용
print("🔧 LoRA 적용 중...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print(f"✅ LoRA 적용 완료")
model.print_trainable_parameters()

# 학습 데이터 로드 (아주 작은 subset)
print("\n📊 데이터 로딩 중...")
with open("/root/data/deepseek_ocr/train_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

# 30개만 사용
train_samples = train_data[:30]
print(f"학습 샘플: {len(train_samples)}개\n")

# 옵티마이저
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# 학습
print("🚀 학습 시작!")
num_epochs = 2
model.train()

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    
    epoch_loss = 0
    processed = 0
    
    for idx, item in enumerate(tqdm(train_samples, desc=f"Epoch {epoch+1}")):
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
            
            # labels 설정
            inputs["labels"] = inputs["input_ids"].clone()
            
            # Forward
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward
            loss.backward()
            
            # 그래디언트 누적 (4 스텝마다 업데이트)
            if (idx + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            processed += 1
            
        except Exception as e:
            print(f"\n⚠️ 샘플 {idx} 처리 실패: {e}")
            continue
    
    # 남은 그래디언트 적용
    optimizer.step()
    optimizer.zero_grad()
    
    avg_loss = epoch_loss / max(processed, 1)
    print(f"\n✅ Epoch {epoch+1} 완료 - 평균 Loss: {avg_loss:.4f}")

# 모델 저장
print("\n💾 모델 저장 중...")
save_dir = "/root/data/qwen2vl_finetuned_mini"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

print(f"✅ 저장 완료: {save_dir}")
print("\n🎉 파인튜닝 완료!")
