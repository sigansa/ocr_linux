#!/usr/bin/env python3
"""
Qwen2-VL 개선된 평가 스크립트
정규화 + 유사도 기반 평가
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json
from tqdm import tqdm
import re
from difflib import SequenceMatcher

print("📦 모델 로딩 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("✅ 모델 로딩 완료\n")

# 텍스트 정규화 함수
def normalize_text(text):
    """텍스트 정규화: 소문자, 공백 제거, 특수문자 정리"""
    # 소문자 변환
    text = text.lower()
    # 공백 제거
    text = re.sub(r'\s+', '', text)
    # 특수문자 제거 (한글, 영문, 숫자만 남김)
    text = re.sub(r'[^가-힣a-z0-9]', '', text)
    return text

# 유사도 계산
def calculate_similarity(text1, text2):
    """두 텍스트의 유사도 계산 (0~1)"""
    return SequenceMatcher(None, text1, text2).ratio()

# 간단한 후처리
def postprocess_prediction(pred_text):
    """불필요한 정보 제거"""
    # 줄바꿈으로 분리된 경우 첫 줄만
    pred_text = pred_text.split('\n')[0].strip()
    
    # "The text on the sign..." 같은 설명 제거
    if 'text' in pred_text.lower() and ('sign' in pred_text.lower() or 'image' in pred_text.lower()):
        # 마지막 따옴표 안의 텍스트 추출 시도
        if '"' in pred_text:
            parts = pred_text.split('"')
            if len(parts) >= 2:
                pred_text = parts[-2]
    
    return pred_text.strip()

# 테스트 데이터 로드
print("📊 테스트 데이터 로딩 중...")
with open("/root/data/deepseek_ocr/val_qwen2vl.jsonl", 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

test_samples = test_data[:20]
print(f"테스트 샘플: {len(test_samples)}개\n")

# 평가
print("🔍 개선된 평가 시작...")
print("="*80)

exact_match = 0
normalized_match = 0
high_similarity = 0  # 유사도 80% 이상
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
    
    # 후처리
    prediction_clean = postprocess_prediction(prediction)
    
    # 정규화
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(prediction_clean)
    
    # 유사도 계산
    similarity = calculate_similarity(gt_norm, pred_norm)
    
    # 정확도 체크
    is_exact = prediction_clean.strip() == ground_truth.strip()
    is_normalized = gt_norm == pred_norm
    is_similar = similarity >= 0.8
    
    if is_exact:
        exact_match += 1
    if is_normalized:
        normalized_match += 1
    if is_similar:
        high_similarity += 1
    
    total += 1
    
    results.append({
        "index": idx + 1,
        "ground_truth": ground_truth,
        "prediction_raw": prediction,
        "prediction_clean": prediction_clean,
        "gt_normalized": gt_norm,
        "pred_normalized": pred_norm,
        "similarity": similarity,
        "exact_match": is_exact,
        "normalized_match": is_normalized,
        "high_similarity": is_similar
    })

# 결과 출력
print("\n" + "="*80)
print("📊 평가 결과")
print("="*80)
print(f"총 샘플: {total}개\n")

print("1️⃣ 완전 일치 (원본 그대로):")
print(f"   정확도: {exact_match/total*100:.2f}% ({exact_match}/{total})")

print("\n2️⃣ 정규화 일치 (소문자+공백제거):")
print(f"   정확도: {normalized_match/total*100:.2f}% ({normalized_match}/{total})")

print("\n3️⃣ 고유사도 (80% 이상):")
print(f"   정확도: {high_similarity/total*100:.2f}% ({high_similarity}/{total})")

print("="*80)

# 샘플 결과 출력
print("\n📝 상세 결과 (처음 10개):")
print("-"*80)
for result in results[:10]:
    print(f"\n샘플 {result['index']}:")
    print(f"  정답: {result['ground_truth']}")
    print(f"  예측: {result['prediction_clean']}")
    print(f"  유사도: {result['similarity']*100:.1f}%")
    
    status = []
    if result['exact_match']:
        status.append("✅ 완전일치")
    if result['normalized_match']:
        status.append("✅ 정규화일치")
    if result['high_similarity']:
        status.append("✅ 고유사도")
    
    if not status:
        status.append("❌ 불일치")
    
    print(f"  상태: {' '.join(status)}")

# 결과 저장
output_file = "/root/data/qwen2vl_improved_eval.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "total": total,
        "exact_match": exact_match,
        "normalized_match": normalized_match,
        "high_similarity": high_similarity,
        "accuracy": {
            "exact": exact_match/total*100,
            "normalized": normalized_match/total*100,
            "similarity_80": high_similarity/total*100
        },
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 결과 저장: {output_file}")
print("\n✅ 개선된 평가 완료!")
