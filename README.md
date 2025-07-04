# VLA_models_training

## 1. gt00t
### 환경 설정
```sh
cd gr00t
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
pip install --upgrade jax jaxlib ml_dtypes
```
> 다음 에러는 무시해도 됨:
"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorboard 2.19.0 requires packaging, which is not installed.
tensorboard 2.19.0 requires six>1.9, which is not installed."
"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gdown 5.2.0 requires beautifulsoup4, which is not installed.
lerobot 0.1.0 requires pyserial>=3.5, which is not installed.
datasets 3.6.0 requires pyarrow>=15.0.0, but you have pyarrow 14.0.1 which is incompatible.
jax 0.6.2 requires ml_dtypes>=0.5.0, but you have ml-dtypes 0.2.0 which is incompatible.
jaxlib 0.6.2 requires ml_dtypes>=0.5.0, but you have ml-dtypes 0.2.0 which is incompatible.
lerobot 0.1.0 requires av>=14.2.0, but you have av 12.3.0 which is incompatible.
lerobot 0.1.0 requires gymnasium==0.29.1, but you have gymnasium 1.0.0 which is incompatible.
lerobot 0.1.0 requires torchvision>=0.21.0, but you have torchvision 0.20.1 which is incompatible.
rerun-sdk 0.23.1 requires pyarrow>=14.0.2, but you have pyarrow 14.0.1 which is incompatible.
tensorstore 0.1.75 requires ml_dtypes>=0.5.0, but you have ml-dtypes 0.2.0 which is incompatible."

### 훈련
#### dataset 경로
* '/demo_data' 아래 데이터 옮기기
* /demo_data/allex_cube/meta/modality.json 있는지 확인!

#### 실행 스크립트
* slurm_ft_allex_bimanual_cube.batch 작성
```sh
#!/bin/bash
#SBATCH --job-name=gr00t-n1.5-ft-allex-bimanual-cube
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t
echo "✅ Conda environment 'gr00t' activated."

mkdir -p tmp 2>/dev/null
mkdir -p checkpoints 2>/dev/null

python scripts/gr00t_finetune.py \
   --dataset-path ./demo_data/allex_cube \
   --num-gpus 1 \
   --output-dir checkpoints/allex-bimanual-cube2  \
   --max-steps 8000 \
   --data-config allex_bimanual_cube \
   --video-backend torchvision_av \
   --action_dim 42 \
   > tmp/slurm-$SLURM_JOB_ID-policy.log 2>&1

# --dataset-path 는 데이터 존재하는 path, 보통 demo_data 아래 있음
# --output-dir 는 생성될 checkpoint 폴더명
# --data-config 는 demo_data아래 있는 폴더명
```

#### 실행
```sh
chmod +X slurm_ft_allex_bimanual_cube.batch	# 실행 권한 부여 (한번만)
sbatch --comment "gr00t training" slurm_ft_allex_bimanual_cube.batch	# slurm을 통한 훈련 실행
squeue --me	# job 돌아가는지 확인
tail -f tmp/slurm-$SLURM_JOB_ID-policy.log	# 로그 확인
```
* checkpoints/ 아래 allex-bimanual-cube2 폴더 생성되는지 확인

### 추론
gpu2 (david@172.30.1.102)에서 설정 및 실행

#### 환경 설정
* gr00t 설치 -> 위 방법 참조

* uv 설치
```sh
curl -Ls https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc
uv --version
```
    
* omni-pilot 설치
git clone git@github.com:RLWRLD/omni-pilot.git
```sh
cd /omni-pilot/packages/gr00t
uv venv
source .venv/bin/activate
cd /omni-pilot
uv pip install setuptools
uv pip install torch
uv pip install psutil
make sync
```

#### 실행
* 학습된 checkpoints 복사 받기
```sh
scp -r david@61.109.237.73:/virtual_lab/rlwrld/david/VLA_models_training/gr00t/checkpoints VLA_models_training/gr00t/artifact
```

* Isaac-GR00T에서 Inference 서버 실행 (터미널 1)
```sh
conda activate gr00t
python scripts/inference_service.py --server --model_path artifact/allex-cube-checkpoints --embodiment_tag new_embodiment --data_config allex_cube --port 7777
```

* omni-pilot에서 robosuite 시뮬레이터 실행 (터미널 2)
```sh
source packages/gr00t/.venv/bin/activate
bin/python packages/gr00t/gr00t_control.py --data-config allex_cube --env-name LiftOurs --task-instruction "Lift the cube from the left stand and place it on the right stand." --episode-horizon 1000 --save-log true --save-video true --no-render --port 7777
```







