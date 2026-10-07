# Morpheus ScoreKit

논문의 평가 파이프라인을 두 개의 독립 모듈로 나눈 구현입니다.

1. `morpheus_scorekit.extraction`: 영상/프레임에서 프레임 정렬 궤적 추출
2. `morpheus_scorekit.scoring`: 유효 궤적만 입력받아 Morpheus 점수 계산

점수 모듈은 추적기에 의존하지 않습니다. 좌표 순서는 공식 코드와 동일한
`[Y(row), X(column), depth]`입니다.

## GPU 서버에서 시작

```bash
git clone --recurse-submodules https://github.com/Bo0sung/morpheus-scorekit.git
cd morpheus-scorekit
bash scripts/setup_gpu_server.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

이미 서버에 clone한 저장소를 갱신할 때는 다음 순서를 사용합니다.

```bash
cd ~/morpheus-scorekit
git pull --ff-only
git submodule update --init --recursive
source .venv/bin/activate
python -m pip install -e ".[sam2]"
export PYTHONPATH="$PWD/src"
python -m morpheus_scorekit.cli run-sam2 -h | grep -- --auto-seed
```

마지막 명령에서 `--auto-seed`가 보여야 현재 CLI입니다.

비공개 저장소이므로 서버에서 GitHub 인증이 필요합니다. `--recurse-submodules`를
빠뜨렸다면 `git submodule update --init --recursive`를 실행합니다. 설치 스크립트는
SAM2 large 체크포인트 하나만 내려받고 연결 상태까지 검사합니다. 서버에서 CUDA 확장
컴파일이 불가능한 경우 `SAM2_BUILD_CUDA=0 bash scripts/setup_gpu_server.sh`로 설치합니다.

## 검증 데이터

공식 데이터셋의 실제 자유낙하 영상 하나만 다운로드합니다.

```powershell
$env:PYTHONPATH = ".deps"
python scripts/download_validation_sample.py
```

현재 샘플의 첫 프레임 클릭 좌표는 공식 `labels.json` 기준 `(642.49, 67.32)`입니다.

### PhyCo-Kubric 강체 자유낙하 데이터

다운로드 위치는 Hugging Face의
[`nnsriram97/phyco_kubric`](https://huggingface.co/datasets/nnsriram97/phyco_kubric)입니다.
강체 공 하나가 플랫폼으로 낙하하고 튀는 데이터는 `ball_drop_v2`이며, 압축 크기는 약
1.3GB입니다. 데이터셋 페이지에서 gated access 조건(연락처 정보 공유)에 먼저 동의한 뒤
로그인합니다.

```powershell
pip install -U huggingface_hub
hf auth login
```

그다음 `ball_drop_v2` 폴더만 다운로드합니다.

```powershell
$env:PYTHONPATH = ".deps"
python scripts/download_phyco_ball_drop_v2.py
```

스크립트를 쓰지 않을 경우 동일한 명령은 다음과 같습니다.

```powershell
hf download nnsriram97/phyco_kubric `
  --repo-type dataset `
  --include "ball_drop_v2/*" `
  --local-dir validation_data/phyco_kubric
```

압축 전체를 풀지 않고 첫 번째 샘플의 RGB·분할·깊이 영상과 메타데이터만 꺼낼 수 있습니다.

```powershell
python scripts/extract_phyco_sample.py `
  validation_data/phyco_kubric/ball_drop_v2/2025-09-04.tar.gz `
  --output validation_data/phyco_sample
```

샘플에는 `rgba.mp4`, `segmentation.mp4`, `depth.mp4`, `metadata.json`,
`animation_data.pkl`이 들어 있습니다. `animation_data.pkl`의 프레임별 world-space 위치·회전은
trajectory predictor의 GT로 쓸 수 있고, RGB에서 추출한 SAM2 궤적의 정확도 검증에도 쓸 수
있습니다. Pickle은 임의 코드를 실행할 수 있으므로 반드시 공식 데이터셋에서 받은 파일만
로드해야 합니다.

추출한 `ball_drop_v2` 샘플의 RGB에서 공을 추적하고, 첫 바닥 접촉 전 자유낙하 구간만 잘라
Morpheus 점수를 계산하려면 다음을 실행합니다.

```bash
EPOCHS=10000 DEVICE=cuda bash scripts/run_phyco_sample.sh \
  validation_data/phyco_sample
```

결과는 `validation_data/phyco_sample/morpheus/scores/combined_scores.json`에 저장됩니다. 이
영상 기반 접촉 프레임은 첫 최저점으로 추정하므로 정적 카메라를 가정합니다. 연구 결과에는
`animation_data.pkl`의 world-space 궤적으로 접촉 시점을 교차 검증해야 합니다. PhyCo 영상은
정상 물리 분포를 측정하는 대조군이며, 물리법칙 위반 민감도는 별도의 Kubric normal/violation
matched pair로 측정합니다. 구체적인 비교 설계는
[`docs/freefall_score_gap_experiment.md`](docs/freefall_score_gap_experiment.md)에 정리했습니다.

공식 PhyCo 분포와 렌더링 방식을 그대로 재생성하려면 저자들의
[`nnsriram97/phyco-sim`](https://github.com/nnsriram97/phyco-sim)을 사용합니다. 반면
Morpheus의 정상/물리위반 점수 민감도를 실험하려면 별도의
`kubric-physics-dataset` 정상/위반 쌍 생성기를 사용합니다.

## 1. 영상 -> 궤적

```powershell
$env:PYTHONPATH = "src"
python -m morpheus_scorekit.cli extract `
  validation_data/real-world-cropped/falling_ball/video_0_fps30/frames_for_tracking `
  --seed 642.4922 67.3234 `
  --fps 30 `
  --experiment falling_ball `
  --output validation_results/falling_ball/trajectory.npz `
  --overlay-dir validation_results/falling_ball/tracking_overlay
```

기본 추출기는 정적 카메라의 통제 실험을 빠르게 검증하기 위한 HSV 기반 추적기입니다.

### 논문 방식: SAM2 + Depth Anything

Windows에서는 코드 의존성을 한 번 설치한 뒤 연결 상태를 확인합니다. 상태 확인 명령은
모델을 내려받거나 로드하지 않습니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_sam2_windows.ps1
```

```powershell
$env:PYTHONPATH = "src"
python -m morpheus_scorekit.cli sam2-doctor
```

`checkpoint_exists`, `config_exists`, `vendored_sam2_exists`, 필요한 Python 모듈이
모두 준비되면 `ready: true`가 됩니다. 기본 체크포인트 위치는
`vendor/Morpheus/checkpoints/sam2.1_hiera_large.pt`입니다. 다른 파일은
`--checkpoint D:/models/sam2.1_hiera_large.pt`로 연결할 수 있습니다. Depth Anything은
기본적으로 `nielsr/depth-anything-large`를 처음 실행할 때 Hugging Face에서 로드하며,
로컬 모델 폴더는 `--depth-model D:/models/depth-anything-large`로 지정합니다.

```powershell
$env:PYTHONPATH = "src"
python -m morpheus_scorekit.cli extract-sam2 `
  validation_data/real-world-cropped/falling_ball/video_0_fps30/frames_for_tracking `
  --seed 1 642.4922 67.3234 `
  --fps 30 `
  --experiment falling_ball `
  --checkpoint D:/models/sam2.1_hiera_large.pt `
  --output validation_results/falling_ball/sam2_trajectory.npz `
  --overlay-dir validation_results/falling_ball/sam2_overlay
```

`--seed` 형식은 `OBJECT_ID X Y`이고 여러 점이나 물체에는 반복 지정합니다. 배경을
제외시키는 클릭은 `--negative OBJECT_ID X Y`입니다. 영상 파일과 프레임 폴더를 모두
받으며, 내부에서 SAM2가 요구하는 `00000.jpg` 구조로 변환합니다.

정적 카메라에서 공 하나가 움직이는 `ball_drop_v2` 영상은 첫 프레임의 움직이는 물체를
자동으로 찾아 seed를 만들 수 있습니다. `auto_seed_preview.jpg`에서 빨간 점이 공 위에
있는지 확인해야 합니다. 틀렸다면 `--auto-seed` 대신 `--seed 1 X Y`를 사용합니다.

모델을 실행하기 전에 자동 seed만 빠르게 확인할 수도 있습니다.

```bash
python -m morpheus_scorekit.cli suggest-seed rgba.mp4 \
  --preview auto_seed_preview.jpg
```

```bash
python -m morpheus_scorekit.cli extract-sam2 rgba.mp4 \
  --auto-seed --seed-preview outputs/auto_seed_preview.jpg \
  --experiment falling_ball --device cuda \
  --output outputs/trajectory.npz \
  --overlay-dir outputs/tracking_overlay
```

### SSH 서버에 영상 하나만 올려 실행

로컬 PC에서 RGB 영상 하나만 복사합니다.

```bash
scp validation_data/phyco_sample/rgba.mp4 USER@HOST:~/morpheus-input/
```

서버에서 저장소와 모델을 최초 한 번 준비합니다. 현재 GitHub 저장소가 비공개라면 clone
시에 GitHub 인증이 필요합니다.

```bash
git clone --recurse-submodules https://github.com/Bo0sung/morpheus-scorekit.git
cd morpheus-scorekit
bash scripts/setup_gpu_server.sh
```

이후에는 영상 경로 하나만 넘기면 자동 seed, SAM2 추적, Depth Anything 깊이 추정을 거쳐
`trajectory.npz`를 만듭니다.

```bash
bash scripts/run_single_video.sh \
  ~/morpheus-input/rgba.mp4 \
  ~/morpheus-output/rgba
```

자동 seed가 잘못되면 첫 프레임에서 확인한 `X Y`를 마지막 두 인자로 지정합니다.

```bash
bash scripts/run_single_video.sh \
  ~/morpheus-input/rgba.mp4 \
  ~/morpheus-output/rgba \
  384 80
```

결과 중 `trajectory.npz`가 좌표 데이터이고, `tracking_overlay/`는 프레임별 추적 확인용,
`auto_seed_preview.jpg`는 자동 선택된 최초 클릭점 확인용입니다. 결과 회수는 다음과 같습니다.

```bash
scp USER@HOST:~/morpheus-output/rgba/trajectory.npz .
scp -r USER@HOST:~/morpheus-output/rgba/tracking_overlay .
```

추적과 점수 계산을 한 번에 실행할 수도 있습니다.

```powershell
python -m morpheus_scorekit.cli run-sam2 `
  validation_data/real-world-cropped/falling_ball/video_0_fps30/frames_for_tracking `
  --seed 1 642.4922 67.3234 `
  --fps 30 --experiment falling_ball `
  --output-dir validation_results/falling_ball/sam2_run `
  --epochs 200000
```

Depth Anything 없이 2D 추적만 확인하려면 `--no-depth`를 붙입니다. 이때 depth 열은
0으로 저장됩니다. 현재 논문의 물리 점수는 이미지 평면 궤적을 사용하므로 점수 실행은
가능하지만, 논문과 같은 전체 추출 파이프라인 검증에는 depth를 켜는 편이 맞습니다.

## 2. 궤적 -> 점수

빠른 검증:

```powershell
$env:PYTHONPATH = "src"
python -m morpheus_scorekit.cli score `
  validation_results/falling_ball/trajectory.npz `
  --experiment falling_ball `
  --output-dir validation_results/falling_ball/scores `
  --epochs 10000
```

논문 설정 재현은 `--epochs 200000`을 사용합니다. 출력의 이름은 공식 코드에 맞춥니다.

- `physical_score`: 논문의 Physical Invariance Score
- `statistical_score`: 논문의 Dynamical Score
- `total_score`: 두 점수의 산술평균

## Kubric normal/violation pair 전체 실행

최신 `kubric-physics-dataset`이 만든 pair에는 Morpheus 호환
`trajectory_freefall_gt.npz`가 포함됩니다. 두 저장소가 같은 상위 폴더에 있을 때 아래 한
명령으로 normal/violation의 GT 점수, 자유낙하 프레임 분리, SAM2 trajectory 추출 및 영상
점수를 모두 실행합니다.

```bash
cd ~/morpheus-scorekit
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

EPOCHS=10000 DEVICE=cuda bash scripts/run_kubric_pair.sh \
  ~/kubric-physics-dataset/outputs/scene_000001
```

결과는 pair 폴더의 `morpheus/` 아래에 저장됩니다.

- `normal_gt/combined_scores.json`, `violation_gt/combined_scores.json`
- `freefall_diagnostics/report.md`: 실제 시간·월드 좌표·동일 프레임 구간 비교
- `freefall_diagnostics/diagnostics.json`: 전체/개입 구간의 수치 결과
- `normal_video/trajectory.npz`, `violation_video/trajectory.npz`
- `normal_video/scores/combined_scores.json`, `violation_video/scores/combined_scores.json`

`freefall_diagnostics`는 Morpheus 점수를 대체하지 않습니다. `state.npz`의 초 단위 시간과
world-space z 좌표를 사용해 정상 중력과의 가속도 오차를 계산하고, normal/violation의 공통
pre-contact 프레임과 개입 프레임을 별도로 비교합니다. 따라서 원본 점수의 평균·시간 정규화로
위반 신호가 얼마나 희석됐는지 함께 제시할 수 있습니다. 이미 GT 점수를 계산했다면 진단만
다시 실행할 수도 있습니다.

```bash
PYTHONPATH=src python scripts/compare_freefall_pair.py \
  ~/kubric-physics-dataset/outputs/scene_000001
```

개입 프레임을 모르는 `state.npz` 하나를 자동 판정하려면 다음을 실행합니다. 각 5프레임
구간의 world-space 위치에 2차식을 fitting하여 가속도를 추정하므로, 저장된 acceleration
배열이나 intervention metadata를 사용하지 않습니다.

```bash
PYTHONPATH=src python scripts/classify_freefall_state.py \
  ~/kubric-physics-dataset/outputs/scene_000001/violation/state.npz \
  --output violation_classifier.json
```

기본 residual 임계값 `0.15`는 동작 확인을 위한 임시값입니다. 실제 실험에서는 정상 validation
데이터의 residual 분포로 이 값을 보정해야 합니다.

## 자유낙하 gravity dose sweep

여러 seed에서 `gravity_scale=0.9, 0.7, 0.5, 0.2, 0.0` pair를 자동 생성하려면 GPU 서버에서
다음을 실행합니다. `kubric-physics-dataset`의 기존 Docker 생성기를 순차 호출하며, 완료된
pair는 재실행 시 건너뜁니다.

```bash
cd ~/morpheus-scorekit
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

GPU_ID=0 python scripts/generate_freefall_dose_sweep.py \
  ~/kubric-physics-dataset \
  --count 20 \
  --start-seed 1000 \
  --doses 0.9 0.7 0.5 0.2 0.0 \
  --start-frame 8 \
  --end-frame 20
```

생성된 모든 pair를 검증하고 dose-response 보고서·JSON·CSV를 만들려면 다음을 실행합니다.

```bash
python scripts/validate_freefall_dose_sweep.py \
  ~/kubric-physics-dataset/outputs/freefall_sweep_manifest.json
```

결과는 `~/kubric-physics-dataset/outputs/freefall_sweep_report/`에 저장됩니다.

- `report.md`: 정확도, F1, 오탐률, dose-response 상관, 배율별 평균 결과
- `sweep_results.json`: 전체 집계 결과
- `per_video_results.csv`: 각 normal/violation 영상의 판정과 peak 구간

검증 코드는 개입 프레임을 classifier 입력으로 사용하지 않습니다. 생성 manifest의 개입 구간은
판정 후 localization IoU를 계산할 때만 사용합니다. 자동 임계값은 정상 GT residual의 중앙값,
MAD, 99 percentile로 계산하며 최소값은 `0.02`입니다. 최종 논문 실험에서는 threshold 보정용
seed와 test seed를 별도로 분리해야 합니다.

## 한 번에 실행

```powershell
$env:PYTHONPATH = "src"
python -m morpheus_scorekit.cli run `
  validation_data/real-world-cropped/falling_ball/video_0_fps30/frames_for_tracking `
  --seed 642.4922 67.3234 `
  --fps 30 `
  --experiment falling_ball `
  --output-dir validation_results/falling_ball `
  --epochs 10000
```

공식 라벨 좌표까지 자동으로 읽는 검증 스크립트도 제공합니다.

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_real_video.py --epochs 10000
```

검증 결과는 `validation_results/falling_ball/validation_summary.json`과
`scores/combined_scores.json`에 저장됩니다.

## 주의

- 물리 현상 이름은 반드시 알려줘야 합니다. 현상마다 ODE와 불변량이 다릅니다.
- 실제 영상은 `real_world_trajectories`, 생성 영상은 CLI의 `--generated`를 사용합니다.
- 실제 영상 점수는 경험적인 상한 기준이며, 최종 점수를 실제 점수로 나눠 정규화하지 않습니다.
- 경량 색상 추적은 자유낙하 검증용입니다. 다물체·외형 변화·가림이 있는 일반 영상에는 SAM2를 사용해야 합니다.
