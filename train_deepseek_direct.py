#!/usr/bin/env python3
"""
DeepSeek-OCR 파인튜닝 스크립트
공식 학습 코드 없이 직접 구현
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from PIL import Image
import json
from tqdm import tqdm
import os
import sys

# DeepSeek-OCR 모델 임포트
sys.path.insert(0, '/root/data/deepseek_ocr_model')
from modeling_deepseekocr import DeepseekOCRForCausalLM, DeepseekOCRConfig

def get_gpu_memory():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB"
    return "N/A"

print("="*70)
print("🎯 DeepSeek-OCR 파인튜닝")
print("="*70)

print(f"\n🔍 초기 GPU 메모리: {get_gpu_memory()}")

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드
print("\n📦 DeepSeek-OCR 모델 로딩 중...")
try:
    model = DeepseekOCRForCausalLM.from_pretrained(
        "/root/data/deepseek_ocr_model",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"✅ 모델 로딩 완료")
    print(f"   GPU 메모리: {get_gpu_memory()}")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    print("\n4비트 없이 재시도...")
    model = DeepseekOCRForCausalLM.from_pretrained(
        "/root/data/deepseek_ocr_model",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"✅ 모델 로딩 완료 (float16)")
    print(f"   GPU 메모리: {get_gpu_memory()}")

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    "/root/data/deepseek_ocr_model",
    trust_remote_code=True,
    local_files_only=True,
)

# LoRA 적용
print("\n🔧 LoRA 적용 중...")
try:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"   GPU 메모리: {get_gpu_memory()}")
except Exception as e:
    print(f"⚠️ LoRA 적용 실패: {e}")
    print("LoRA 없이 진행...")

# 데이터 로드 (매우 작은 샘플)
print("\n📊 데이터 로딩 중...")
with open("/root/data/deepseek_ocr/train.jsonl", 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

train_samples = train_data[:3]  # 3개만
print(f"학습 샘플: {len(train_samples)}개\n")

# 옵티마이저
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

# 학습
print("🚀 학습 시작!")
print("-"*70)

model.train()
total_loss = 0
success_count = 0

for idx, item in enumerate(train_samples):
    try:
        print(f"\n[샘플 {idx+1}/{len(train_samples)}]")
        print(f"  학습 전 메모리: {get_gpu_memory()}")
        
        # 데이터 추출
        image_path = item['image']
        conversations = item['conversations']
        
        # 대화에서 텍스트 추출
        user_text = conversations[0]['content']
        assistant_text = conversations[1]['content']
        
        print(f"  정답: {assistant_text}")
        
        # 이미지 로드
        image = Image.open(f"/root/data/deepseek_ocr/{image_path}").convert("RGB")
        
        # DeepSeek-OCR 입력 형식
        prompt = f"USER: <image>\n{user_text}\nASSISTANT: {assistant_text}"
        
        # 전처리 (DeepSeek-OCR의 process 메서드 사용)
        inputs = model.build_conversation_input_ids(
            tokenizer,
            query=user_text,
            images=[image],
            history=[],
        )
        
        # GPU로 이동
        input_ids = inputs['input_ids'].to(model.device)
        pixel_values = inputs['pixel_values'].to(model.device)
        images_seq_mask = inputs['images_seq_mask'].to(model.device)
        images_spatial_crop = inputs['images_spatial_crop']
        
        print(f"  입력 처리 후: {get_gpu_memory()}")
        
        # labels 설정
        labels = input_ids.clone()
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
        )
        
        loss = outputs.loss
        
        print(f"  Forward 후: {get_gpu_memory()}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward
        loss.backward()
        
        print(f"  Backward 후: {get_gpu_memory()}")
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        success_count += 1
        
        # 메모리 정리
        del inputs, input_ids, pixel_values, images_seq_mask, outputs, loss
        torch.cuda.empty_cache()
        
        print(f"  정리 후 메모리: {get_gpu_memory()}")
        print(f"  ✅ 성공")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OOM 에러!")
        print(f"  메모리: {get_gpu_memory()}")
        torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"  ❌ 에러: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*70)
if success_count > 0:
    avg_loss = total_loss / success_count
    print(f"✅ 학습 완료!")
    print(f"   - 성공: {success_count}/{len(train_samples)}")
    print(f"   - 평균 Loss: {avg_loss:.4f}")
    print(f"   - 최종 메모리: {get_gpu_memory()}")
    
    # 모델 저장
    print("\n💾 모델 저장 중...")
    save_dir = "/root/data/deepseek_ocr_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"✅ 저장 완료: {save_dir}")
    except Exception as e:
        print(f"⚠️ 저장 실패: {e}")
else:
    print(f"❌ 모든 샘플 실패")

print("="*70)
