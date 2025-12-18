#!/usr/bin/env python3
"""
DeepSeek-OCR 파인튜닝 스크립트 (Unsloth 사용)
한국어 간판 OCR 데이터셋으로 파인튜닝
"""

import os
import json
from unsloth import FastVisionModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from PIL import Image
import torch

# GPU 메모리 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_dataset_from_jsonl(jsonl_path, images_base_dir):
    """
    JSONL 파일에서 데이터셋 로드
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            image_path = os.path.join(images_base_dir, item['image'])
            
            if not os.path.exists(image_path):
                continue
            
            # DeepSeek-OCR 형식으로 변환
            conversations = item['conversations']
            user_msg = conversations[0]['content']
            assistant_msg = conversations[1]['content']
            
            data.append({
                'image': image_path,
                'prompt': f"<image>\n{user_msg}",
                'answer': assistant_msg
            })
    
    return Dataset.from_list(data)


def format_data(examples):
    """
    데이터를 DeepSeek-OCR 학습 형식으로 변환
    """
    texts = []
    images = []
    
    for i in range(len(examples['prompt'])):
        # 학습 형식: <image>\n질문 -> 답변
        text = f"{examples['prompt'][i]}\n{examples['answer'][i]}"
        texts.append(text)
        
        # 이미지 로드
        try:
            img = Image.open(examples['image'][i]).convert('RGB')
            images.append(img)
        except:
            images.append(None)
    
    return {
        'text': texts,
        'image': images
    }


def main():
    print("="*60)
    print("DeepSeek-OCR 한국어 간판 파인튜닝")
    print("="*60)
    
    # 설정
    TRAIN_JSONL = "/root/data/deepseek_ocr/train.jsonl"
    VAL_JSONL = "/root/data/deepseek_ocr/val.jsonl"
    TRAIN_IMAGES_DIR = "/root/data/deepseek_ocr/filtered_train"
    VAL_IMAGES_DIR = "/root/data/deepseek_ocr/filtered_val"
    OUTPUT_DIR = "/root/data/deepseek_ocr_finetuned"
    
    # 1. 모델 로드
    print("\n[1/5] 모델 로딩 중...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/DeepSeek-OCR",
        load_in_4bit=True,  # 4bit 양자화로 메모리 절약
        use_gradient_checkpointing="unsloth",  # 메모리 효율적인 학습
    )
    
    # 2. LoRA 설정
    print("[2/5] LoRA 어댑터 추가 중...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,      # Vision encoder도 파인튜닝
        finetune_language_layers=True,     # Language model도 파인튜닝
        finetune_attention_modules=True,   # Attention 모듈 파인튜닝
        finetune_mlp_modules=True,         # MLP 모듈 파인튜닝
        
        r=16,           # LoRA rank
        lora_alpha=16,  # LoRA alpha
        lora_dropout=0, # LoRA dropout
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # 3. 데이터셋 로드
    print("[3/5] 데이터셋 로딩 중...")
    train_dataset = load_dataset_from_jsonl(TRAIN_JSONL, TRAIN_IMAGES_DIR)
    val_dataset = load_dataset_from_jsonl(VAL_JSONL, VAL_IMAGES_DIR)
    
    print(f"  - Training: {len(train_dataset):,} 샘플")
    print(f"  - Validation: {len(val_dataset):,} 샘플")
    
    # 데이터 포맷팅
    train_dataset = train_dataset.map(
        format_data,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        format_data,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 4. 학습 설정
    print("[4/5] 학습 설정 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,      # GPU 메모리에 맞게 조정
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,      # 효과적인 batch size = 8
        warmup_steps=50,
        max_steps=500,                      # 총 학습 스텝 (조정 가능)
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        optim="adamw_8bit",                 # 8bit optimizer로 메모리 절약
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",                   # wandb 없이 로컬 로깅
    )
    
    # 5. Trainer 설정 및 학습
    print("[5/5] 학습 시작...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    
    # 학습 시작
    trainer.train()
    
    # 6. 모델 저장
    print("\n모델 저장 중...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print("\n" + "="*60)
    print("✅ 파인튜닝 완료!")
    print(f"모델 저장 위치: {OUTPUT_DIR}/final_model")
    print("="*60)


if __name__ == "__main__":
    main()
