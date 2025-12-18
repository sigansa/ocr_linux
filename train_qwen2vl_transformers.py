#!/usr/bin/env python3
"""
Qwen2-VL 파인튜닝 스크립트 (기본 Transformers 사용)
한국어 간판 OCR 프로젝트
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from PIL import Image
import json

print("📦 패키지 로딩 완료")

# 모델과 프로세서 로드
print("🔄 모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("✅ 모델 로딩 완료")

# LoRA 설정
print("🔧 LoRA 설정 중...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 데이터셋 로드
print("📊 데이터셋 로딩 중...")
train_dataset = load_dataset("json", data_files="/root/data/deepseek_ocr/train_qwen2vl.jsonl", split="train")
val_dataset = load_dataset("json", data_files="/root/data/deepseek_ocr/val_qwen2vl.jsonl", split="train")

print(f"- 학습 샘플: {len(train_dataset)}개")
print(f"- 검증 샘플: {len(val_dataset)}개")

# 데이터 전처리 함수
def preprocess_function(examples):
    """데이터를 모델 입력 형식으로 변환"""
    batch = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}
    
    for messages in examples["messages"]:
        # 이미지와 텍스트 추출
        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]
        
        # 이미지 경로와 텍스트 분리
        image_path = user_content[0]["image"]
        user_text = user_content[1]["text"]
        assistant_text = assistant_content[0]["text"]
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"이미지 로드 실패: {image_path}")
            continue
        
        # 대화 형식으로 구성
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
        
        # 프로세서로 인코딩
        text = processor.apply_chat_template(conversation, tokenize=False)
        inputs = processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        batch["input_ids"].append(inputs["input_ids"][0])
        batch["attention_mask"].append(inputs["attention_mask"][0])
        batch["labels"].append(inputs["input_ids"][0])
        
        if "pixel_values" in inputs:
            batch["pixel_values"].append(inputs["pixel_values"][0])
        if "image_grid_thw" in inputs:
            batch["image_grid_thw"].append(inputs["image_grid_thw"][0])
    
    # 텐서로 변환
    if len(batch["input_ids"]) > 0:
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        batch["labels"] = torch.stack(batch["labels"])
        
        if len(batch["pixel_values"]) > 0:
            batch["pixel_values"] = torch.stack(batch["pixel_values"])
        if len(batch["image_grid_thw"]) > 0:
            batch["image_grid_thw"] = torch.stack(batch["image_grid_thw"])
    
    return batch

print("🔄 데이터 전처리 중...")
# 샘플만 사용 (메모리 절약)
train_dataset_small = train_dataset.select(range(min(100, len(train_dataset))))
val_dataset_small = val_dataset.select(range(min(20, len(val_dataset))))

# 학습 설정
training_args = TrainingArguments(
    output_dir="/root/data/qwen2vl_korean_signboard_v2",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
)

print("🚀 학습 시작!")
print(f"- 학습 샘플: {len(train_dataset_small)}개")
print(f"- 검증 샘플: {len(val_dataset_small)}개")
print(f"- 배치 크기: {training_args.per_device_train_batch_size}")
print(f"- Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"- 총 effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# 간단한 데이터 콜레이터
def collate_fn(examples):
    """배치 데이터 결합"""
    return preprocess_function({"messages": [ex["messages"] for ex in examples]})

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_small,
    eval_dataset=val_dataset_small,
    data_collator=collate_fn,
)

# 학습 시작
trainer_stats = trainer.train()

# 모델 저장
print("\n💾 모델 저장 중...")
model.save_pretrained("/root/data/qwen2vl_korean_signboard_v2/final")
processor.save_pretrained("/root/data/qwen2vl_korean_signboard_v2/final")

print("\n✅ 학습 완료!")
print(f"- 학습 Loss: {trainer_stats.training_loss:.4f}")
print(f"- 저장 위치: /root/data/qwen2vl_korean_signboard_v2/final")
