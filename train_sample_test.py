#!/usr/bin/env python3
"""
DeepSeek-OCR 샘플 테스트 학습 스크립트
1000개 샘플로 빠른 테스트
"""

import os
import json
from unsloth import FastVisionModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from PIL import Image
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_dataset_from_jsonl(jsonl_path, images_base_dir):
    """JSONL 파일에서 데이터셋 로드"""
    data = []
    skipped = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            image_path = os.path.join(images_base_dir, item['image'])
            
            if not os.path.exists(image_path):
                skipped += 1
                continue
            
            conversations = item['conversations']
            user_msg = conversations[0]['content']
            assistant_msg = conversations[1]['content']
            
            data.append({
                'image': image_path,
                'prompt': f"<image>\n{user_msg}",
                'answer': assistant_msg
            })
    
    print(f"  로드: {len(data)}개, 스킵: {skipped}개")
    return Dataset.from_list(data)


def format_data(examples):
    """데이터를 DeepSeek-OCR 학습 형식으로 변환"""
    texts = []
    images = []
    
    for i in range(len(examples['prompt'])):
        text = f"{examples['prompt'][i]}\n{examples['answer'][i]}"
        texts.append(text)
        
        try:
            img = Image.open(examples['image'][i]).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"이미지 로드 실패: {examples['image'][i]}")
            images.append(None)
    
    return {
        'text': texts,
        'image': images
    }


def main():
    print("="*60)
    print("DeepSeek-OCR 샘플 테스트 학습")
    print("="*60)
    
    # 설정
    TRAIN_JSONL = "/root/data/deepseek_ocr/train_sample.jsonl"
    VAL_JSONL = "/root/data/deepseek_ocr/val_sample.jsonl"
    TRAIN_IMAGES_DIR = "/root/data/deepseek_ocr/filtered_train"
    VAL_IMAGES_DIR = "/root/data/deepseek_ocr/filtered_val"
    OUTPUT_DIR = "/root/data/deepseek_ocr_test"
    
    # 1. 모델 로드
    print("\n[1/5] 모델 로딩 중 (HuggingFace에서 다운로드)...")
    try:
        from transformers import AutoModel
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/DeepSeek-OCR",  # 온라인에서 직접 로드
            load_in_4bit=True,
            auto_model=AutoModel,
            trust_remote_code=True,
            use_gradient_checkpointing="unsloth",
        )
        print("  ✓ 모델 로드 완료")
    except Exception as e:
        print(f"  ✗ 모델 로드 실패: {e}")
        return
    
    # 2. LoRA 설정
    print("[2/5] LoRA 어댑터 추가 중...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("  ✓ LoRA 설정 완료")
    
    # 3. 데이터셋 로드
    print("[3/5] 데이터셋 로딩 중...")
    train_dataset = load_dataset_from_jsonl(TRAIN_JSONL, TRAIN_IMAGES_DIR)
    val_dataset = load_dataset_from_jsonl(VAL_JSONL, VAL_IMAGES_DIR)
    
    print(f"  - Training: {len(train_dataset):,} 샘플")
    print(f"  - Validation: {len(val_dataset):,} 샘플")
    
    # 데이터 포맷팅
    print("  데이터 포맷팅 중...")
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
    print("  ✓ 데이터 준비 완료")
    
    # 4. 학습 설정
    print("[4/5] 학습 설정 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,                      # 테스트용으로 50 스텝만
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )
    print("  ✓ 학습 설정 완료")
    
    # 5. Trainer 설정 및 학습
    print("[5/5] Trainer 초기화 중...")
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
    print("  ✓ Trainer 준비 완료")
    
    # 학습 시작
    print("\n" + "="*60)
    print("학습 시작...")
    print("="*60)
    
    try:
        trainer.train()
        print("\n✅ 학습 완료!")
        
        # 모델 저장
        print("\n모델 저장 중...")
        model.save_pretrained(f"{OUTPUT_DIR}/final_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
        print(f"✓ 모델 저장: {OUTPUT_DIR}/final_model")
        
    except Exception as e:
        print(f"\n✗ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("테스트 완료")
    print("="*60)


if __name__ == "__main__":
    main()
