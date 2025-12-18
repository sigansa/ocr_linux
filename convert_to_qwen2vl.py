#!/usr/bin/env python3
"""
Qwen2-VL 파인튜닝용 데이터 변환
기존 DeepSeek 형식을 Qwen2-VL 형식으로 변환
"""

import json
import os

def convert_to_qwen2vl_format(input_jsonl, output_jsonl, images_base_dir):
    """
    DeepSeek 형식을 Qwen2-VL 형식으로 변환
    
    Qwen2-VL 형식:
    {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": "path"}, {"type": "text", "text": "질문"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "답변"}]}
        ]
    }
    """
    converted_data = []
    skipped = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            image_path = os.path.join(images_base_dir, item['image'])
            
            if not os.path.exists(image_path):
                skipped += 1
                continue
            
            conversations = item['conversations']
            user_msg = conversations[0]['content']
            assistant_msg = conversations[1]['content']
            
            # Qwen2-VL 형식으로 변환
            qwen_format = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_msg}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_msg}
                        ]
                    }
                ]
            }
            converted_data.append(qwen_format)
    
    # JSONL 형식으로 저장
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"변환 완료: {len(converted_data)}개")
    print(f"스킵: {skipped}개")
    print(f"출력: {output_jsonl}")


if __name__ == "__main__":
    # Training 데이터 변환
    print("=== Training 데이터 변환 ===")
    convert_to_qwen2vl_format(
        input_jsonl="/root/data/deepseek_ocr/train_sample.jsonl",
        output_jsonl="/root/data/deepseek_ocr/train_qwen2vl.jsonl",
        images_base_dir="/root/data/deepseek_ocr/filtered_train"
    )
    
    # Validation 데이터 변환
    print("\n=== Validation 데이터 변환 ===")
    convert_to_qwen2vl_format(
        input_jsonl="/root/data/deepseek_ocr/val_sample.jsonl",
        output_jsonl="/root/data/deepseek_ocr/val_qwen2vl.jsonl",
        images_base_dir="/root/data/deepseek_ocr/filtered_val"
    )
    
    print("\n✅ 데이터 변환 완료!")
