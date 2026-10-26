# Subject 9 Machine Vision

OpenCV 기반 제조 머신비전 검사기술을 단계적으로 학습하는 프로젝트입니다.

---

## Day 01 — 머신비전 검사문제 설계와 ROI

1일차에는 완성된 OK / NG 검사기를 만드는 것이 아니라
이미지와 영상에서 **어디를 검사할 것인지 정하고,
정상과 불량의 차이를 어떻게 측정 가능한 특징으로 바꿀 것인지**
생각하는 과정을 학습했습니다.

### 핵심 흐름

```text
검사 문제 확인
    ↓
이미지 읽기
    ↓
ROI 설정
    ↓
여러 이미지에 같은 ROI 적용
    ↓
영상 Frame 처리
    ↓
영상 Frame에 ROI 적용
    ↓
스마트폰 실시간 영상 연결
    ↓
실시간 Frame에 ROI 적용
    ↓
새로운 검사 문제에 ROI 직접 적용
    ↓
결과와 판단 기록
```

### 만든 코드

| 파일 | 역할 |
|---|---|
| `src/00_check_env.py` | Python · OpenCV · NumPy 환경 확인 |
| `src/01_read_image.py` | 이미지 한 장 읽기 → `outputs/read_image_check.jpg` |
| `src/02_image_info.py` | 이미지 크기와 채널 확인 |
| `src/03_roi_crop.py` | 이미지 한 장에서 ROI 설정·저장 |
| `src/04_batch_roi.py` | 여러 이미지에 동일한 ROI 적용 |
| `src/05_video_input.py` | 저장 영상을 Frame으로 읽고 30 Frame마다 저장 |
| `src/06_video_roi.py` | 저장 영상의 Frame에 ROI 적용·저장 |
| `src/07_webcam_input.py` | 스마트폰 IP Webcam 영상 입력 확인 |
| `src/08_phone_camera_roi.py` | 스마트폰 실시간 영상에서 마우스로 ROI 선택·저장 |
| `src/03_roi_crop_challenge.py` | Mini Challenge — `03`의 경로·좌표만 바꾼 복사본 |
| `src/04_batch_roi_challenge.py` | Mini Challenge — `04`의 경로·좌표만 바꾼 복사본 |
| `03_roi_crop_review.ipynb` | `03_roi_crop.py`를 STEP 1~41로 나눠 이해하는 Jupyter 리뷰 |

### 확인한 결과

| 실습 | ROI (x1, y1, x2, y2) | 결과 |
|---|---|---|
| 병뚜껑 한 장 (`03`) | (170, 120, 470, 450) | ROI (330, 300, 3) — `sample01` 뚜껑 전체 포함 |
| 병뚜껑 여러 장 (`04`) | (170, 120, 470, 490) | 5 / 5 처리. 1차(y2=450)는 `sample02` 뚜껑 아래가 잘려 아래쪽 여유 추가 |
| 저장 영상 (`05`·`06`) | (430, 90, 760, 260) | 1280×720, 24 FPS, 240 Frame → 8장 저장. Frame 30은 병이 들어오는 중이라 실패 |
| 스마트폰 실시간 (`07`·`08`) | 마우스 선택 | 스마트폰·Wi-Fi 필요 — 코드만 준비. 강사 결과 예시 `outputs/live_roi/cap_*_reference.jpg` |
| Mini Challenge (왕관 뚜껑) | v1 (500, 110, 740, 230) → v2 (430, 90, 760, 260) | v1은 Frame 60에서 뚜껑 왼쪽이 잘림 → v2에서 해결 |

자세한 관찰과 판단: [`reports/day01_problem_definition.md`](reports/day01_problem_definition.md) ·
데이터 역할: [`reports/dataset_roles.csv`](reports/dataset_roles.csv)

### 오늘의 핵심

```text
한 장에서 잘 맞는 ROI ≠ 모든 이미지에서 잘 맞는 ROI

ROI가 제품을 놓쳤다      → 검사 영역 문제
ROI는 정상인데 뚜껑이 없다 → 실제 불량 후보
```

### 실행 방법

```bash
cd ~/ai_vision/subject9_machinevision
source .venv/bin/activate
python src/04_batch_roi.py
```

Jupyter: `./jupyter.sh` (또는 `jupyter notebook --no-browser --port=8888`)

### 다음 학습

2일차 — Image Enhancement: 오늘 정한 ROI 안에서 검사하기 좋은 영상 상태를 만드는 방법

---

## Day 02 — Image Enhancement와 검사 영상 품질 개선

1일차에 설정한 ROI를 이용하여
촬영조건과 영상 품질이 검사 결과에 어떤 영향을 주는지 확인했다.

기준 이미지: 강사 제공 `samples/day02/cap_on_reference.jpg` (1일차 17절 실시간 ROI 결과 대신 사용)

### 실습 내용

| 파일 | 내용 | 결과 |
|---|---|---|
| `src/09_color_space.py` | BGR / Gray / HSV 비교 | Gray·V가 병뚜껑-배경 차이를 가장 잘 보여줌. H는 채도가 낮아 불안정 |
| `src/10_brightness_contrast.py` | Brightness / Contrast (beta 0·20·50·80) | beta 20~50이 자연스러움. 80은 뿌옇게 됨 |
| `src/11_gamma.py` | Gamma 0.6~2.0 | 1.2~1.5가 어두운 홈을 살리면서 덜 뿌옇다 |
| `src/12_clahe.py` | CLAHE clipLimit 1~4 | 2.0이 적절. 3.0 이상은 배경 Texture도 강조 |
| `src/13_filter_compare.py` | Gaussian / Median / Bilateral | 점 Noise에는 Median 3이 가장 적절 |
| `src/14_sharpen.py` | Sharpening | 흐린 홈을 일부 복원, 경계 주변이 약간 거칠어짐 |
| `src/15_threshold_otsu.py` | Threshold 80·120·160 / Otsu | 80·160은 실패, Otsu가 118을 자동 선택 |
| `src/16_preprocess_compare.py` | 전처리 전·후 Otsu 비교 | Gaussian + Otsu가 가장 깨끗함, CLAHE + Otsu는 조각이 늘어남 |
| `src/17_day02_challenge.py` | Mini Challenge | 전처리 미사용 + 3일차 추가 확인 (아래) |
| `roi_temp/crop_challenge.py` | 원본 challenge 사진(4000×3000)을 마우스로 ROI 자르기 | 강사 제공 ROI 결과(943×897)를 사용 |

### 핵심 결과

- `outputs/day02/compare/preprocess_compare.png`
- `outputs/day02/challenge/challenge_result.jpg` · `challenge_trials_compare.png`
- `reports/day02_parameter_table.csv`
- `reports/day02_preprocess_reason.md`

### Mini Challenge 판단

흰 병뚜껑과 회색 배경이 모두 밝은 계열(배경 134~147 ≥ 병뚜껑 131)이라
Brightness · Gamma · CLAHE 어떤 것도 Otsu에서 배경을 분리하지 못했다.
→ **전처리 미사용**, 원인은 화질이 아니라 촬영조건(배경·조명). 3일차에서 어두운 경계선을 Contour로 확인한다.

### 오늘의 핵심

전처리 기술을 먼저 선택하는 것이 아니라

문제 확인
→ 필요한 전처리 선택
→ Parameter 비교
→ Threshold / Otsu 결과 확인
→ 사용 여부 결정

순서로 판단한다.

### 다음 학습

Day 03에서는 Binary 이미지에
Morphology와 Contour를 적용하여
객체의 면적·개수·위치를 측정하고
Rule 기반 OK / NG 판정으로 연결한다.
