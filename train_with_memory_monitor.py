#!/usr/bin/env python3
"""
Qwen2-VL 파인튜닝 with 메모리 모니터링
"""

import torch
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from PIL import Image
import json
from tqdm import tqdm
import os

def get_gpu_memory():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB"
    return "N/A"

def clear_memory():
    """메모리 정리"""
    gc.collect()
    torch.cuda.empty_cache()

print("="*70)
print("🎯 Qwen2-VL 파인튜닝 with 메모리 모니터링")
print("="*70)

# 초기 메모리
print(f"🔍 초기 GPU 메모리: {get_gpu_memory()}")

# 메모리 정리
clear_memory()
print(f"🧹 정리 후: {get_gpu_memory()}")

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
    local_files_only=True,  # 캐시만 사용
)
print(f"   GPU 메모리: {get_gpu_memory()}")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    local_files_only=True,  # 캐시만 사용
)

# LoRA 적용
print("\n🔧 LoRA 적용 중...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f"   GPU 메모리: {get_gpu_memory()}")

# 데이터 로드 (5개만)
print("\n📊 데이터 로딩 중...")
with open("/root/data/deepseek_ocr/train_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

train_samples = train_data[:5]  # 5개만
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
        
        messages = item["messages"]
        
        # 데이터 추출
        image_path = messages[0]["content"][0]["image"]
        user_text = messages[0]["content"][1]["text"]
        assistant_text = messages[1]["content"][0]["text"]
        
        print(f"  정답: {assistant_text}")
        
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
        
        print(f"  입력 처리 후: {get_gpu_memory()}")
        
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Forward
        outputs = model(**inputs)
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
        del inputs, outputs, loss, image
        clear_memory()
        
        print(f"  정리 후 메모리: {get_gpu_memory()}")
        print(f"  ✅ 성공")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OOM 에러!")
        print(f"  메모리: {get_gpu_memory()}")
        clear_memory()
        print(f"  정리 후: {get_gpu_memory()}")
        continue
    except Exception as e:
        print(f"  ❌ 에러: {e}")
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
    save_dir = "/root/data/qwen2vl_tiny_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    print(f"✅ 저장 완료: {save_dir}")
else:
    print(f"❌ 모든 샘플 실패")

print("="*70)
