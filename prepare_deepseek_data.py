#!/usr/bin/env python3
"""
DeepSeek OCR 파인튜닝용 데이터셋 생성
- 이미지-텍스트 쌍을 JSONL 형식으로 변환
- DeepSeek VL 모델이 요구하는 형식에 맞춤
"""

import json
import os
from pathlib import Path
from tqdm import tqdm


def create_deepseek_dataset(
    labels_json: str,
    images_dir: str,
    output_jsonl: str,
    use_relative_path: bool = True
):
    """
    DeepSeek VL 파인튜닝용 데이터셋 생성
    
    Args:
        labels_json: 필터링된 labels.json 경로
        images_dir: 이미지 디렉토리 경로
        output_jsonl: 출력 JSONL 파일 경로
        use_relative_path: 상대 경로 사용 여부
    """
    
    print(f"JSON 로딩: {labels_json}")
    with open(labels_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    print(f"전체 이미지: {len(images):,}개")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    converted = 0
    skipped = 0
    
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for img in tqdm(images, desc="데이터셋 생성"):
            korean_name = img.get('store_kor_name', '').strip()
            english_name = img.get('store_eng_name', '').strip()
            
            # 간판명 조합 (한글 우선, 영문 보조)
            if korean_name and english_name:
                signboard_text = f"{korean_name} ({english_name})"
            elif korean_name:
                signboard_text = korean_name
            elif english_name:
                signboard_text = english_name
            else:
                skipped += 1
                continue
            
            # 이미지 경로
            if use_relative_path:
                image_path = os.path.join('images', img['fileName'])
            else:
                image_path = os.path.join(images_dir, img['fileName'])
            
            # 이미지 파일 존재 확인
            full_path = os.path.join(images_dir, img['fileName'])
            if not os.path.exists(full_path):
                skipped += 1
                continue
            
            # DeepSeek VL 형식으로 변환
            # 간단한 OCR 태스크: "이미지에서 간판 텍스트를 읽어주세요"
            entry = {
                "image": image_path,
                "conversations": [
                    {
                        "role": "user",
                        "content": "이 간판 이미지의 텍스트를 읽어주세요."
                    },
                    {
                        "role": "assistant",
                        "content": signboard_text
                    }
                ],
                "metadata": {
                    "business_category": img.get('business_category', ''),
                    "signboard_type": img.get('item_sub_category', ''),
                    "location": img.get('location', '')
                }
            }
            
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            converted += 1
    
    print(f"\n=== 변환 완료 ===")
    print(f"변환된 데이터: {converted:,}개")
    print(f"스킵된 데이터: {skipped:,}개")
    print(f"출력 파일: {output_jsonl}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek OCR 파인튜닝 데이터셋 생성')
    parser.add_argument('--labels', type=str, required=True, help='필터링된 labels.json 경로')
    parser.add_argument('--images', type=str, required=True, help='이미지 디렉토리 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 JSONL 파일 경로')
    parser.add_argument('--absolute-path', action='store_true', help='절대 경로 사용')
    
    args = parser.parse_args()
    
    create_deepseek_dataset(
        labels_json=args.labels,
        images_dir=args.images,
        output_jsonl=args.output,
        use_relative_path=not args.absolute_path
    )
