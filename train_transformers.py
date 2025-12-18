#!/usr/bin/env python3
"""
DeepSeek-OCR Transformers 방식 파인튜닝
"""

import os
import json
import torch
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings('ignore')

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def load_jsonl_data(jsonl_path, images_base_dir):
    """JSONL 데이터 로드"""
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
                'image_path': image_path,
                'prompt': f"<image>\n{user_msg}",
                'text': assistant_msg,
            })
    
    print(f"  로드: {len(data)}개, 스킵: {skipped}개")
    return Dataset.from_list(data)


class DeepSeekOCRDataCollator:
    """데이터 collator"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # 간단한 배치 처리
        batch = {
            'image_paths': [f['image_path'] for f in features],
            'prompts': [f['prompt'] for f in features],
            'texts': [f['text'] for f in features],
        }
        return batch


class DeepSeekOCRTrainer(Trainer):
    """커스텀 Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """간단한 더미 loss (실제로는 모델의 forward 사용)"""
        # DeepSeek-OCR은 특수한 학습 방식이 필요하므로
        # 여기서는 간단한 테스트만 진행
        loss = torch.tensor(0.5, requires_grad=True)
        
        if return_outputs:
            return loss, None
        return loss


def main():
    print("="*60)
    print("DeepSeek-OCR Transformers 파인튜닝")
    print("="*60)
    
    # 설정
    TRAIN_JSONL = "/root/data/deepseek_ocr/train_sample.jsonl"
    VAL_JSONL = "/root/data/deepseek_ocr/val_sample.jsonl"
    TRAIN_IMAGES_DIR = "/root/data/deepseek_ocr/filtered_train"
    VAL_IMAGES_DIR = "/root/data/deepseek_ocr/filtered_val"
    MODEL_PATH = "/root/data/deepseek_ocr_model"
    OUTPUT_DIR = "/root/data/deepseek_ocr_transformers"
    
    # 1. 모델 로드
    print("\n[1/5] 모델 로딩 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # 8bit 양자화로 메모리 절약
        )
        print("  ✓ 모델 로드 완료")
    except Exception as e:
        print(f"  ✗ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. LoRA 설정
    print("[2/5] LoRA 적용 중...")
    try:
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("  ✓ LoRA 적용 완료")
    except Exception as e:
        print(f"  ⚠️  LoRA 적용 실패 (모델이 지원하지 않을 수 있음): {e}")
        print("  → LoRA 없이 계속 진행...")
    
    # 3. 데이터 로드
    print("[3/5] 데이터셋 로딩 중...")
    train_dataset = load_jsonl_data(TRAIN_JSONL, TRAIN_IMAGES_DIR)
    val_dataset = load_jsonl_data(VAL_JSONL, VAL_IMAGES_DIR)
    
    print(f"  - Training: {len(train_dataset):,} 샘플")
    print(f"  - Validation: {len(val_dataset):,} 샘플")
    
    # 4. 학습 설정
    print("[4/5] 학습 설정 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )
    
    # 5. Trainer 설정
    print("[5/5] Trainer 초기화 중...")
    
    data_collator = DeepSeekOCRDataCollator(model, tokenizer)
    
    trainer = DeepSeekOCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("\n" + "="*60)
    print("⚠️  중요 공지")
    print("="*60)
    print("DeepSeek-OCR은 일반적인 Transformers 학습과 다릅니다.")
    print("실제 파인튜닝을 위해서는:")
    print("1. DeepSeek-OCR의 공식 학습 코드 필요")
    print("2. 또는 Unsloth Colab 노트북 사용")
    print("3. 또는 Qwen2-VL 같은 표준 모델 사용")
    print("="*60)
    
    # 테스트 실행
    print("\n현재는 구조 테스트만 진행합니다...")
    print(f"모델이 로드되었고, 데이터가 준비되었습니다.")
    print(f"실제 학습을 위해서는 DeepSeek의 공식 학습 코드가 필요합니다.")
    
    # 모델 저장 테스트
    print(f"\n모델 저장 테스트: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✓ Tokenizer 저장 완료")
    
    print("\n" + "="*60)
    print("테스트 완료")
    print("="*60)


if __name__ == "__main__":
    main()
