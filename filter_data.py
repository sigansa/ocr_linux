#!/usr/bin/env python3
"""
간판 OCR 데이터셋 필터링 스크립트
- 한글/영문 간판명이 모두 없는 데이터 제거
- 이미지 파일이 실제로 존재하는지 확인
- 흐릿한 이미지 제거 (선택사항)
- 필터링된 이미지를 새 디렉토리로 복사
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm


def is_blurry(image_path: str, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    라플라시안 분산을 사용해 이미지가 흐릿한지 판별
    
    Args:
        image_path: 이미지 파일 경로
        threshold: 흐릿함 판별 임계값 (낮을수록 흐림)
    
    Returns:
        (is_blurry, variance): 흐릿한지 여부와 분산값
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return True, 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var < threshold, laplacian_var
    except Exception as e:
        print(f"이미지 읽기 오류 ({image_path}): {e}")
        return True, 0.0


def filter_dataset(
    json_path: str,
    images_dir: str,
    output_dir: str,
    check_blur: bool = False,
    blur_threshold: float = 100.0,
    min_text_length: int = 1,
    copy_images: bool = True
):
    """
    데이터셋 필터링 및 유효한 이미지 복사
    
    Args:
        json_path: 입력 JSON 파일 경로
        images_dir: 이미지 디렉토리 경로
        output_dir: 출력 디렉토리 (이미지와 JSON이 저장될 폴더)
        check_blur: 흐릿한 이미지 체크 여부
        blur_threshold: 흐릿함 판별 임계값
        min_text_length: 최소 텍스트 길이
        copy_images: 이미지 파일 복사 여부
    """
    
    print(f"JSON 파일 로딩: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    total_count = len(images)
    
    # 출력 디렉토리 생성
    output_images_dir = os.path.join(output_dir, 'images')
    output_json_path = os.path.join(output_dir, 'labels.json')
    
    if copy_images:
        os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== 필터링 시작 ===")
    print(f"전체 데이터: {total_count:,}개")
    print(f"이미지 디렉토리: {images_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    filtered_images = []
    stats = {
        'no_text': 0,
        'file_not_found': 0,
        'too_short_text': 0,
        'blurry': 0,
        'korean_only': 0,
        'english_only': 0,
        'both_languages': 0,
        'valid': 0
    }
    
    for img in tqdm(images, desc="필터링 진행"):
        # 1. 한글/영문 간판명 체크
        korean_name = img.get('store_kor_name', '').strip()
        english_name = img.get('store_eng_name', '').strip()
        
        # 둘 다 없으면 제외
        if not korean_name and not english_name:
            stats['no_text'] += 1
            continue
        
        # 텍스트 통계
        if korean_name and english_name:
            stats['both_languages'] += 1
        elif korean_name:
            stats['korean_only'] += 1
        else:
            stats['english_only'] += 1
        
        # 2. 텍스트 길이 체크 (최소 한쪽이라도 충족해야 함)
        korean_valid = len(korean_name) >= min_text_length if korean_name else False
        english_valid = len(english_name) >= min_text_length if english_name else False
        
        if not korean_valid and not english_valid:
            stats['too_short_text'] += 1
            continue
        
        # 3. 이미지 파일 존재 체크
        image_path = os.path.join(images_dir, img['fileName'])
        if not os.path.exists(image_path):
            stats['file_not_found'] += 1
            continue
        
        # 4. 흐릿한 이미지 체크 (선택사항)
        if check_blur:
            is_blur, variance = is_blurry(image_path, blur_threshold)
            if is_blur:
                stats['blurry'] += 1
                continue
        
        # 필터 통과 - 이미지 복사
        if copy_images:
            dest_path = os.path.join(output_images_dir, img['fileName'])
            shutil.copy2(image_path, dest_path)
        
        filtered_images.append(img)
        stats['valid'] += 1
    
    # 결과 저장
    output_data = {
        'images': filtered_images,
        'annotations': []  # bbox annotations 제거
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 통계 출력
    print(f"\n=== 필터링 결과 ===")
    print(f"전체: {total_count:,}개")
    print(f"\n제거된 데이터:")
    print(f"  - 한글/영문 모두 없음: {stats['no_text']:,}개")
    print(f"  - 이미지 파일 없음: {stats['file_not_found']:,}개")
    print(f"  - 텍스트 너무 짧음: {stats['too_short_text']:,}개")
    if check_blur:
        print(f"  - 흐릿한 이미지: {stats['blurry']:,}개")
    
    print(f"\n유효한 데이터: {stats['valid']:,}개 ({stats['valid']/total_count*100:.1f}%)")
    print(f"  - 한글만: {stats['korean_only']:,}개")
    print(f"  - 영문만: {stats['english_only']:,}개")
    print(f"  - 한글+영문: {stats['both_languages']:,}개")
    
    if copy_images:
        print(f"\n이미지 복사 완료: {output_images_dir}")
    print(f"JSON 저장 완료: {output_json_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='간판 OCR 데이터셋 필터링 및 복사')
    parser.add_argument('--json', type=str, required=True, help='입력 JSON 파일 경로')
    parser.add_argument('--images', type=str, required=True, help='이미지 디렉토리 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 디렉토리 경로')
    parser.add_argument('--check-blur', action='store_true', help='흐릿한 이미지 체크')
    parser.add_argument('--blur-threshold', type=float, default=100.0, help='흐릿함 임계값')
    parser.add_argument('--min-text-length', type=int, default=1, help='최소 텍스트 길이')
    parser.add_argument('--no-copy-images', action='store_true', help='이미지 복사 안함 (JSON만 생성)')
    
    args = parser.parse_args()
    
    filter_dataset(
        json_path=args.json,
        images_dir=args.images,
        output_dir=args.output,
        check_blur=args.check_blur,
        blur_threshold=args.blur_threshold,
        min_text_length=args.min_text_length,
        copy_images=not args.no_copy_images
    )
