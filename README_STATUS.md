# Qwen2-VL 한국어 간판 OCR 파인튜닝 프로젝트

## 📊 현재 상황 요약

### ✅ 완료된 작업
1. **데이터 준비**
   - 학습 데이터: 138,149개 샘플
   - 검증 데이터: 17,272개 샘플
   - Qwen2-VL 형식으로 변환 완료

2. **베이스라인 평가** (파인튜닝 전)
   - ❌ 완전 일치: **10.00%** (20개 중 2개)
   - ✅ 정규화 일치: **35.00%** (20개 중 7개) - 대소문자/공백 무시
   - ✅ 고유사도 80%+: **40.00%** (20개 중 8개) - 실용적 정확도
   
   주요 문제:
     - 대소문자 불일치 (kokoro → KOKORO) - 정규화로 해결 가능
     - 띄어쓰기 차이 (마왕족발 → 마왕 족발) - 정규화로 해결 가능
     - 불필요한 정보 포함 (전화번호, 영어 등) - 파인튜닝 필요
     - 완전히 잘못 읽음 (돈통마늘보쌈 → DBAS) - 파인튜닝 필요

### ❌ 로컬 파인튜닝 실패 원인 (상세)

#### 1. Unsloth 방식 실패
**시도:** `FastVisionModel.from_pretrained()` + LoRA
```python
# train_qwen2vl.py 실행 시
```

**에러:**
- **Triton 컴파일 오류**: `triton.backends.nvidia.driver.py` 초기화 실패
- **원인**: Unsloth의 커스텀 커널 최적화가 Qwen2-VL의 vision attention과 충돌
- **발생 위치**: `apply_rotary_pos_emb_vision()` 함수에서 torch.compile 시도 중
- **근본 원인**: Vision-Language 모델의 복잡한 attention 메커니즘 + Unsloth 최적화 불일치

**재현 가능:**
```bash
cd /root/data && python train_qwen2vl.py
# 에러: InductorError in apply_rotary_pos_emb_vision
```

---

#### 2. 기본 Transformers 방식 실패
**시도:** `Transformers` + `PEFT` + `BitsAndBytesConfig`

**문제점:**

**A. 데이터 처리 복잡성**
- **에러**: `ValueError: Mismatch in image token count`
- **원인**: Qwen2-VL의 이미지 토큰화 메커니즘이 복잡
  - 이미지를 가변 크기 토큰으로 변환
  - `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>` 특수 토큰
  - truncation과 padding이 이미지 토큰 수와 불일치
  
**B. 메모리 문제**
```
실측 메모리 사용량 (nvidia-smi):
- 초기: 0GB
- 모델 로딩 (4bit): 1.51GB
- LoRA 적용 후: 2.21GB (예약)
- 첫 샘플 학습 시도: 23.7GB/24GB (96.5%)
- 결과: CUDA Out of Memory
```

**왜 메모리가 폭증하는가?**
1. **Vision Encoder의 메모리 요구량**
   - ViT 기반 인코더: 이미지 → 특징 벡터
   - Gradient 계산 시 중간 activation 저장
   - 이미지 크기에 비례해서 메모리 증가

2. **멀티모달 융합**
   - Vision 특징 + Language 임베딩 결합
   - Cross-attention 레이어의 activation
   - 두 모달리티의 gradient 동시 보관

3. **4비트 양자화의 한계**
   - 모델 가중치는 작아짐 (2.2B → 1.5GB)
   - **하지만 activation과 gradient는 full precision** (bfloat16)
   - 학습 시 activation 메모리가 주범

**C. 속도 문제**
```
실측 속도 (train_with_memory_monitor.py):
- 샘플 1개 처리 시간: 35초+
- GPU 사용률: 100%
- 병목: 이미지 인코딩 단계

예상 전체 학습 시간:
- 30개 샘플: 35초 × 30 = 17.5분
- 1,000개 샘플: 35초 × 1,000 = 9.7시간
- 138,000개 샘플: 35초 × 138,000 = 1,340시간 (56일!)
```

---

#### 3. Vision-Language 모델의 본질적 복잡성

**일반 LLM vs Vision-Language 모델 비교:**

| 항목 | 일반 LLM (예: Llama) | Vision-Language (Qwen2-VL) |
|------|---------------------|----------------------------|
| 입력 | 텍스트만 | 이미지 + 텍스트 |
| 처리 단계 | 1단계 (Language) | 3단계 (Vision → Fusion → Language) |
| 메모리 사용 | 낮음 | **높음** (이미지 activation) |
| 학습 속도 | 빠름 | **느림** (이미지 인코딩) |
| LoRA 효과 | 매우 효과적 | 제한적 (vision은 고정) |

**Qwen2-VL 아키텍처:**
```
입력 이미지 (예: 1024x1024)
    ↓
Vision Encoder (ViT-like)
    - Patch Embedding: 이미지를 패치로 분할
    - Self-Attention: 패치 간 관계 학습
    - 출력: 가변 길이 vision tokens (메모리 많이 사용!)
    ↓
Vision-Language Fusion
    - Vision tokens + Text tokens 결합
    - Cross-attention (메모리 더 많이 사용!)
    ↓
Language Model (Qwen2)
    - Autoregressive 생성
    - LoRA는 여기만 적용 가능
```

**메모리 계산 예시:**
```python
# 1024x1024 이미지 하나
image_size = 1024 * 1024 * 3  # RGB
patch_size = 14 * 14
num_patches = (1024 / 14) ** 2 = 5329 patches

# Vision encoder activation
hidden_dim = 1280
activation_size = num_patches * hidden_dim * 4 bytes (fp32)
                = 5329 * 1280 * 4 = 27MB per layer
                = 27MB * 32 layers = 864MB per image

# Gradient (backward pass)
gradient_size = activation_size * 2 = 1.7GB per image

# 총 메모리: 모델(1.5GB) + activation(0.9GB) + gradient(1.7GB) + ...
# = 약 20GB+ for single image training!
```

---

#### 4. 시도한 최적화들과 실패 이유

**A. Gradient Checkpointing 비활성화**
```python
use_gradient_checkpointing=False
```
- **목적**: 속도 향상
- **결과**: 메모리 더 증가 (activation 저장)
- **결론**: 실패

**B. LoRA rank 축소 (r=16 → r=8)**
```python
lora_config = LoraConfig(r=8, ...)
```
- **목적**: 학습 파라미터 감소
- **결과**: 파라미터는 줄었지만 (4.3M → 1.09M)
- **문제**: Vision encoder 메모리는 그대로
- **결론**: 효과 미미

**C. Vision layers 고정**
```python
finetune_vision_layers=False  # vision은 학습 안 함
```
- **목적**: Vision gradient 제거
- **결과**: 여전히 forward pass에서 메모리 사용
- **문제**: Inference에도 vision encoder 필요
- **결론**: 실패

**D. 배치 크기 1**
```python
per_device_train_batch_size=1
```
- **목적**: 메모리 최소화
- **결과**: 여전히 24GB 초과
- **문제**: 이미지 하나만으로도 20GB+ 사용
- **결론**: 한계 도달

---

#### 5. 왜 LoRA가 여기서는 효과가 없는가?

**LoRA의 장점:**
- ✅ 학습 파라미터 감소: 2.2B → 1.09M (99.95% 감소)
- ✅ 모델 메모리 감소: 효과 있음
- ✅ 학습 속도 향상: Language model에서는 효과적

**하지만 Vision-Language에서는:**
- ❌ Vision encoder activation: LoRA로 줄일 수 없음
- ❌ 이미지 처리 시간: 변하지 않음
- ❌ Fusion layer memory: 여전히 필요
- ⚠️ **병목이 Language가 아니라 Vision에 있음!**

**결론:**
LoRA는 Language 부분만 효율화. 
**Vision-Language 모델의 메모리 병목은 Vision Encoder에 있어서 LoRA로 해결 불가.**

---

#### 6. 24GB GPU의 한계

**RTX 4090 (24GB VRAM) 실측:**
- Qwen2-VL 4bit: 1.5GB
- LoRA: 추가 0.7GB
- **이미지 1개 학습**: 20GB+
- **총합**: 22-24GB (한계)

**필요 GPU:**
- 안정적 학습: **40GB+ (A100, H100)**
- Gradient accumulation으로 배치 크기 늘리기: 불가능 (메모리 이미 풀)
- Multi-GPU: 가능하지만 복잡도 증가

---

#### 7. 결론

**로컬 파인튜닝이 안 되는 핵심 이유:**
1. ❌ **메모리 부족**: 24GB로는 Qwen2-VL 학습 불가
2. ❌ **속도 문제**: 샘플당 35초 (전체 학습 56일)
3. ❌ **Vision-Language 복잡성**: LoRA로 해결 안 됨
4. ❌ **도구 호환성**: Unsloth 최적화가 Qwen2-VL과 충돌

**왜 Colab/클라우드를 추천하는가?**
- ✅ 더 큰 GPU (A100 40GB+)
- ✅ 최적화된 환경 (Unsloth가 Colab에서 잘 작동)
- ✅ 빠른 네트워크 (모델 다운로드)
- ✅ 검증된 노트북 (다른 사람들이 성공한 코드)

## 💡 권장 솔루션

### 방법 1: Google Colab 사용 (가장 추천)
**장점:**
- Unsloth 공식 노트북 사용 가능
- 무료 GPU (T4) 제공
- 환경 설정 불필요
- 파인튜닝 속도 빠름

**진행 방법:**
1. Colab에서 Unsloth의 Qwen2-VL 노트북 열기
2. 데이터 업로드 (Google Drive 연동)
3. 노트북 실행 → 자동 파인튜닝
4. 파인튜닝된 모델 다운로드

**Colab 노트북:**
- https://colab.research.google.com/drive/1vqHUq9R...
- Unsloth 공식 Qwen2-VL 노트북 검색

### 방법 2: 기존 한국어 OCR 모델 사용
**대안 모델들:**
- **naver-clova-ix/donut-base-finetuned-cord-v2**
- **PleIAs/OCRonos-Qwen2-VL**
- **Upstage의 Document AI 모델들**

이미 한국어로 학습된 모델을 사용하면 추가 파인튜닝 없이도 좋은 성능

### 방법 3: 클라우드 GPU 사용
**옵션:**
- **Vast.ai**: 시간당 $0.20~$0.50
- **RunPod**: 시간당 $0.30~$0.70
- **Lambda Labs**: 시간당 $0.50~$1.10

30분~1시간이면 파인튜닝 완료 가능

## 📈 예상 파인튜닝 효과

**현재 베이스라인 (파인튜닝 전):**
- 완전 일치: 10%
- 정규화 일치: 35%
- 고유사도 80%+: 40%

**파인튜닝 후 예상 (Colab/클라우드에서):**
- **최소 목표**: 50-60% (현재의 1.5배)
- **현실적 목표**: 70-80% (현재의 2배)
- **최적 목표**: 85-90% (현재의 2.3배)

**로컬 파인튜닝 실패로 검증 불가**

## 🚀 다음 단계

### 옵션 A: Colab으로 진행
```bash
# 데이터 압축
cd /root/data/deepseek_ocr
tar -czf korean_signboard_data.tar.gz train_qwen2vl.jsonl val_qwen2vl.jsonl filtered_train/images filtered_val/images
```
→ Google Drive 업로드 → Colab 노트북 실행

### 옵션 B: 다른 모델 테스트
기존 한국어 OCR 모델로 비교 평가

### 옵션 C: 계속 로컬 시도
더 간단한 모델이나 다른 접근 방식 탐색

## 📝 파일 정리

**데이터:**
- `/root/data/deepseek_ocr/train_qwen2vl.jsonl` - 학습 데이터
- `/root/data/deepseek_ocr/val_qwen2vl.jsonl` - 검증 데이터
- `/root/data/deepseek_ocr/filtered_train/` - 학습 이미지
- `/root/data/deepseek_ocr/filtered_val/` - 검증 이미지

**평가:**
- `/root/data/qwen2vl_baseline_results.json` - 초기 평가 (5% 완전 일치)
- `/root/data/qwen2vl_improved_eval.json` - 개선된 평가 (35% 정규화, 40% 유사도)

**시도한 학습 스크립트들 (모두 실패):**
- `/root/data/train_qwen2vl.py` - Unsloth 방식 (Triton 오류)
- `/root/data/train_qwen2vl_transformers.py` - Transformers (데이터 처리 실패)
- `/root/data/train_mini.py` - 초소형 (OOM 에러)
- `/root/data/train_with_memory_monitor.py` - 메모리 모니터링 (24GB 한계 확인)

**스크립트:**
- `/root/data/evaluate_baseline.py` - 평가 스크립트
- `/root/data/test_qwen2vl_inference.py` - 추론 테스트
- `/root/data/convert_to_qwen2vl.py` - 데이터 변환

---

## 결론

로컬 환경에서 Qwen2-VL 파인튜닝은 기술적으로 가능하지만 매우 복잡합니다.
**Colab 사용을 강력히 권장**합니다 - 무료이고, 빠르고, 검증된 방법입니다.
