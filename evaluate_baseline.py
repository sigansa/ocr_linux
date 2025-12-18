#!/usr/bin/env python3
"""
Qwen2-VL 파인튜닝 전 모델 평가
여러 샘플로 성능 테스트
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json
from tqdm import tqdm

print("📦 모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("✅ 모델 로딩 완료\n")

# 테스트 데이터 로드
print("📊 테스트 데이터 로딩 중...")
with open("/root/data/deepseek_ocr/val_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

# 20개 샘플만 테스트
test_samples = test_data[:20]
print(f"테스트 샘플: {len(test_samples)}개\n")

# 평가 시작
print("🔍 모델 평가 시작...")
print("="*80)

correct = 0
total = 0
results = []

for idx, item in enumerate(tqdm(test_samples, desc="평가 중")):
    messages = item["messages"]
    
    # 정답 추출
    image_path = messages[0]["content"][0]["image"]
    user_text = messages[0]["content"][1]["text"]
    ground_truth = messages[1]["content"][0]["text"]
    
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
        }
    ]
    
    # 추론
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # 응답 추출
    if "assistant\n" in generated_text:
        prediction = generated_text.split("assistant\n")[-1].strip()
    else:
        prediction = generated_text.split("assistant")[-1].strip() if "assistant" in generated_text else generated_text
    
    # 정확도 계산 (완전 일치)
    is_correct = prediction.strip() == ground_truth.strip()
    if is_correct:
        correct += 1
    total += 1
    
    results.append({
        "index": idx + 1,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": is_correct
    })

# 결과 출력
print("\n" + "="*80)
print("📊 평가 결과")
print("="*80)
print(f"총 샘플: {total}개")
print(f"정확히 맞춘 샘플: {correct}개")
print(f"정확도: {correct/total*100:.2f}%")
print("="*80)

# 샘플 결과 출력
print("\n📝 샘플 결과 (처음 10개):")
print("-"*80)
for result in results[:10]:
    status = "✅" if result["correct"] else "❌"
    print(f"\n{status} 샘플 {result['index']}:")
    print(f"   정답: {result['ground_truth']}")
    print(f"   예측: {result['prediction']}")

# 결과 저장
output_file = "/root/data/qwen2vl_baseline_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "total": total,
        "correct": correct,
        "accuracy": correct/total*100,
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 결과 저장: {output_file}")
print("\n✅ 평가 완료!")
