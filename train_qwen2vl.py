#!/usr/bin/env python3
"""
Qwen2-VL 파인튜닝 스크립트 (Unsloth 사용)
한국어 간판 OCR 프로젝트
"""

import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 4비트 양자화로 모델 로드
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing=False,  # gradient checkpointing 비활성화
)

# LoRA 어댑터 추가 (language layers만 파인튜닝)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # vision은 고정
    finetune_language_layers=True,  # language만 학습
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# 데이터셋 로드
train_dataset = load_dataset("json", data_files="/root/data/deepseek_ocr/train_qwen2vl.jsonl", split="train")
val_dataset = load_dataset("json", data_files="/root/data/deepseek_ocr/val_qwen2vl.jsonl", split="train")

print(f"✅ 데이터셋 로드 완료")
print(f"- 학습 샘플: {len(train_dataset)}개")
print(f"- 검증 샘플: {len(val_dataset)}개")

# 데이터 콜레이터 설정
data_collator = UnslothVisionDataCollator(model, tokenizer)

# SFT 트레이너 설정
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        
        warmup_steps=50,
        num_train_epochs=1,
        max_steps=100,  # 샘플 테스트용으로 100 스텝만
        
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        
        logging_steps=10,
        eval_steps=25,
        save_steps=25,
        save_total_limit=3,
        
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        
        output_dir="/root/data/qwen2vl_korean_signboard",
        report_to="none",
        
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
    ),
)

# 학습 시작
print("\n🚀 학습 시작!")
trainer_stats = trainer.train()

# 모델 저장
print("\n💾 모델 저장 중...")
model.save_pretrained("/root/data/qwen2vl_korean_signboard/final")
tokenizer.save_pretrained("/root/data/qwen2vl_korean_signboard/final")

print("\n✅ 학습 완료!")
print(f"- 학습 Loss: {trainer_stats.training_loss:.4f}")
print(f"- 총 스텝: {trainer_stats.global_step}")
print(f"- 저장 위치: /root/data/qwen2vl_korean_signboard/final")

# 간단한 추론 테스트
print("\n🧪 추론 테스트...")
FastVisionModel.for_inference(model)

# 첫 번째 검증 샘플로 테스트
test_sample = val_dataset[0]
test_messages = test_sample["messages"]
test_image = test_messages[0]["content"][0]["image"]
test_question = test_messages[0]["content"][1]["text"]
correct_answer = test_messages[1]["content"][0]["text"]

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": test_image}, {"type": "text", "text": test_question}]}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
generated = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, 
                          use_cache=True, temperature=0.5, min_p=0.1)

print(f"\n정답: {correct_answer}")
