import torch
import numpy as np
import json
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from pathlib import Path
import tqdm

try:
    from rlwrld_deployment import UniVLAInference
except ImportError:
    print("="*50)
    print("오류: 'rlwrld_deployment.py'에서 'UniVLAInference' 클래스를 찾을 수 없습니다.")
    print("두 스크립트(inference_rlwrld.py와 rlwrld_deployment.py)가 동일한 폴더에 있는지 확인해주세요.")
    print("="*50)
    exit()

# ==============================================================================
# TODO: 사용자 설정 영역
# ==============================================================================
################################################################################
### allex
# # 1. 훈련이 완료된 "실험 폴더" 경로
# # 예시: "runs/univla-7b+rlwrld_custom--test_state_action_filter=w-LowLevelDecoder-ws-12"
# EXPERIMENT_PATH = "runs/univla-7b+real_world+b32+lr-0.00035+lora-r32+dropout-0.0--allex_state_action_filter=w-LowLevelDecoder-ws-12"
# # 2. 훈련된 체크포인트 스텝 번호
# CHECKPOINT_STEP = 20000
# # 3. 테스트에 사용할 데이터가 있는 폴더 경로
# TEST_DATA_ROOT = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset_multiview_converted_state_action"

# # 4. (★★★★★ 중요) 훈련 스크립트와 ★완벽하게 동일하게★ 설정해야 하는 인덱스
# ### 원본 데이터셋 그대로 사용하여 훈련
# # CHECKOUT_NAME = "allex_all"
# # INDICES_FOR_STATE = list(range(60))
# # INDICES_FOR_ACTION = list(range(42))

# ### 데이터에서 오른팔 및 손 움직임만 훈련하기 위해 인덱스 설정
# INDICES_FOR_STATE = list(range(4)) + list(range(6, 13)) + list(range(20, 40)) 
# INDICES_FOR_ACTION = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
################################################################################

################################################################################
### gr1
# 1. 훈련이 완료된 "실험 폴더" 경로
# 예시: "runs/univla-7b+rlwrld_custom--test_state_action_filter=w-LowLevelDecoder-ws-12"
EXPERIMENT_PATH = "runs/univla-7b+real_world+b32+lr-0.00035+lora-r32+dropout-0.0--gr1_state_action_filter=w-LowLevelDecoder-ws-12"
# 2. 훈련된 체크포인트 스텝 번호
CHECKPOINT_STEP = 10000
# 3. 테스트에 사용할 데이터가 있는 폴더 경로
TEST_DATA_ROOT = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/gr1-cube-dataset_multiview_converted_state_action"

### 원본 데이터셋 그대로 사용하여 훈련
# INDICES_FOR_STATE = list(range(42))
# INDICES_FOR_ACTION = list(range(24))

### 데이터에서 오른팔 및 손 움직임만 훈련하기 위해 인덱스 설정
INDICES_FOR_STATE = list(range(13)) + list(range(20, 31))
INDICES_FOR_ACTION = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
################################################################################

STATE_DIM = len(INDICES_FOR_STATE)
ACTION_DIM = len(INDICES_FOR_ACTION)
WINDOW_SIZE = 12

# 5. 평가할 에피소드 수 (None이면 전체 에피소드를 평가합니다)
NUM_EPISODES_TO_EVAL = 5
# ==============================================================================

def main():
    # 경로 조합
    SAVED_VLA_PATH = f"{EXPERIMENT_PATH}/{CHECKPOINT_STEP}"
    DECODER_PATH = f"{SAVED_VLA_PATH}/action_decoder-{CHECKPOINT_STEP}.pt"
    STATS_PATH = f"{EXPERIMENT_PATH}/dataset_statistics.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"추론에 사용할 디바이스: {device}")

    # 추론 엔진 초기화
    print("추론 엔진을 초기화합니다...")
    policy = UniVLAInference(
        saved_model_path=SAVED_VLA_PATH,
        pred_action_horizon=WINDOW_SIZE,
        decoder_path=DECODER_PATH,
        device=device,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM 
    )
    print("추론 엔진 초기화 완료.")

    # 정규화 통계 로드
    try:
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        print(f"- 정규화 통계 파일 로드 완료: {STATS_PATH}")
        state_mean = np.array(stats['state_mean'])
        state_std = np.array(stats['state_std'])
        action_mean = np.array(stats['action_mean'])
        action_std = np.array(stats['action_std'])
    except Exception as e:
        print(f"[오류] 정규화 통계 파일 로딩 실패: {e}")
        return

####################### 오직 하나의 dataset에 대해 평가하는 코드
    # 테스트 데이터 준비
    print("\n테스트용 데이터를 준비합니다...")
    test_episode_path = Path(TEST_DATA_ROOT) / "episode_000000"
    
    current_image_pil = Image.open(test_episode_path / "frame_001.png").convert("RGB")
    
    with open(test_episode_path / "instruction.txt", "r") as f:
        task_instruction = f.read().strip()

    current_state_full = np.load(test_episode_path / "state.npy")[0]
    proprio_filtered = current_state_full[INDICES_FOR_STATE]
    proprio_normalized = (proprio_filtered - state_mean[INDICES_FOR_STATE]) / state_std[INDICES_FOR_STATE]
    proprio_tensor = torch.from_numpy(proprio_normalized).float().to(device)
    
    print(f"- 언어 지시어: '{task_instruction}'")
    print("- 이미지 및 State 준비 완료.")

    # 추론 실행
    print("\n추론을 시작합니다...")
    with torch.no_grad():
        predicted_actions_normalized = policy.step(current_image_pil, task_instruction, proprio_tensor)
    
    # 결과 역정규화 및 확인
    print("\n--- 추론 결과 ---")
    # pred_actions_numpy = predicted_actions_normalized.squeeze(0).cpu().numpy()
    pred_actions_numpy = predicted_actions_normalized.squeeze(0).cpu().float().numpy()
    unnormalized_actions = (pred_actions_numpy * action_std[INDICES_FOR_ACTION]) + action_mean[INDICES_FOR_ACTION]
    
    print(f"실제 행동 값(역정규화됨, 처음 5개 스텝, {ACTION_DIM}차원):")
    print(unnormalized_actions[:5, :])

####################### 여러 episode, step에 대한 dataset에 대해 평가하는 코드
    all_episode_paths = sorted([p for p in Path(TEST_DATA_ROOT).iterdir() if p.is_dir()])
    episode_paths_to_eval = all_episode_paths[:NUM_EPISODES_TO_EVAL] if NUM_EPISODES_TO_EVAL is not None else all_episode_paths

    print(f"\n총 {len(episode_paths_to_eval)}개의 에피소드에 대해 평가를 시작합니다.")
    
    all_errors = []

    for episode_path in tqdm.tqdm(episode_paths_to_eval, desc="Episodes"):
        all_states_full = np.load(episode_path / "state.npy")
        all_actions_full = np.load(episode_path / "action.npy")
        with open(episode_path / "instruction.txt", "r") as f:
            task_instruction = f.read().strip()
        
        policy.reset(task_instruction)
        
        num_frames = len(all_states_full)
        
        for frame_idx in tqdm.trange(num_frames - WINDOW_SIZE, desc=f"  Frames in {episode_path.name}", leave=False):
            image_path = episode_path / f"frame_{frame_idx + 1:03d}.png"
            current_image_pil = Image.open(image_path).convert("RGB")
            
            current_state_full = all_states_full[frame_idx]
            proprio_filtered = current_state_full[INDICES_FOR_STATE]
            proprio_normalized = (proprio_filtered - state_mean[INDICES_FOR_STATE]) / state_std[INDICES_FOR_STATE]
            proprio_tensor = torch.from_numpy(proprio_normalized).float().to(device)
            
            with torch.no_grad():
                predicted_actions_normalized = policy.step(current_image_pil, task_instruction, proprio_tensor)
            
            # ★★★ 핵심 수정: .numpy() 호출 전에 .float()으로 타입을 변환합니다. ★★★
            pred_actions_numpy = predicted_actions_normalized.squeeze(0).cpu().float().numpy()
            
            unnormalized_actions = (pred_actions_numpy * action_std[INDICES_FOR_ACTION]) + action_mean[INDICES_FOR_ACTION]
            
            ground_truth_actions_window = all_actions_full[frame_idx : frame_idx + WINDOW_SIZE]
            ground_truth_actions_filtered = ground_truth_actions_window[:, INDICES_FOR_ACTION]
            
            error = np.linalg.norm(unnormalized_actions[0] - ground_truth_actions_filtered[0])
            all_errors.append(error)

    if all_errors:
        average_error = np.mean(all_errors)
        print("\n--- 최종 평가 결과 ---")
        print(f"평가된 총 스텝 수: {len(all_errors)}")
        print(f"예측과 정답 사이의 평균 L2 오차: {average_error:.4f}")
    else:
        print("\n평가를 수행할 데이터가 없습니다.")

if __name__ == "__main__":
    main()