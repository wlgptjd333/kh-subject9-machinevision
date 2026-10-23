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
