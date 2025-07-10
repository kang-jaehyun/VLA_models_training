# vla-scripts/rlwrld_deployment.py

from typing import Optional
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
# from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.policy.transformer_utils import MAPBlock

# Register OpenVLA model to HF Auto Classes
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# ==============================================================================
# 훈련 코드와 ★완벽하게 동일한★ ActionDecoderHead 클래스
# ==============================================================================
ACTION_HISTORY_LEN = 5

class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size, hidden_dim, state_dim, action_dim):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, window_size * action_dim)
        )
        self.hist_proj = nn.Sequential(
            nn.Flatten(),                                        # (B,H,A) → (B,H*A)
            nn.Linear(ACTION_HISTORY_LEN * action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, latent_action_tokens, visual_embed, proprio, action_hist):
        latent_action_tokens = latent_action_tokens[:, -4:]
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        hist_feat = self.hist_proj(action_hist)
        concat = torch.cat(
            [self.attn_pool(latent_action_tokens, init_embed=visual_embed),
             proprio,
             hist_feat],
            dim=-1,
        )
        action = self.proj(concat)
        return action

# ==============================================================================
# 최종 UniVLAInference 클래스
# ==============================================================================
class UniVLAInference:
    def __init__(
        self,
        saved_model_path: str,
        decoder_path: str,
        pred_action_horizon: int,
        state_dim: int,
        action_dim: int,
        device: str = 'cuda',
        norm_stats: dict = None
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = torch.device(device)
        self.pred_action_horizon = pred_action_horizon
        self.norm_stats = norm_stats

        # 1. VLA 본체 모델 로드
        print(f"[*] VLA 모델 로딩: {saved_model_path}")
        self.vla = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()

        # 2. Processor 로드
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        
        # 3. ActionDecoderHead 생성 및 가중치 로드
        print(f"[*] Action Decoder 로딩: {decoder_path}")
        # 3-1. CPU에서 모델 구조를 먼저 생성합니다.
        self.action_decoder = ActionDecoderHead(
            window_size=pred_action_horizon,
            hidden_dim=512, 
            state_dim=state_dim,
            action_dim=action_dim
        )
        # 3-2. CPU에서 훈련된 가중치를 불러옵니다.
        self.action_decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
        
        # 3-3. 가중치가 로드된 모델을 최종적으로 GPU로 보내고, 데이터 타입을 bfloat16으로 변환합니다.
        self.action_decoder = self.action_decoder.to(device=self.device, dtype=torch.bfloat16)
        
        # 3-4. 추론 모드로 설정합니다.
        self.action_decoder.eval()

        # 4. Prompt Builder 준비
        self.prompt_builder = PurePromptBuilder("openvla")
        self.reset("Placeholder Task")
        self.prev_hist_action = ['']

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        # 필요한 다른 리셋 로직이 있다면 여기에 추가 (예: action history)
        
    def step(
        self, image: Image.Image, task_description: str, proprio: torch.Tensor, action_hist: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            image: PIL Image 객체
            task_description: str; "Lift the cube"
            proprio: torch.Tensor; (1, STATE_DIM) 크기의 정규화된 state 텐서
        Output:
            action: torch.Tensor; (1, window_size, ACTION_DIM) 크기의 정규화된 예측 액션
        """
        if task_description != self.task_description:
            self.reset(task_description)

        # 1. 프롬프트 생성
        prompt_builder = PurePromptBuilder("openvla")
        if len(self.prev_hist_action[-1]) > 0:
            prompt = f"In: What action should the robot take to {task_description.lower()}? History action {self.prev_hist_action[-1]}\nOut:"
        else:
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

        conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # 2. 입력 전처리 (VLA는 bfloat16을 기대)
        inputs = self.processor(
            text=prompt_builder.get_prompt(),
            images=image,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        # 3. VLA 순전파 (Latent Action 예측)
        with torch.no_grad():
            latent_action_tokens, visual_embed, generated_ids = self.vla.predict_latent_action(
                **inputs,
                # unnorm_key=self.unnorm_key,      # 필요 없다면 제거
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
            )
            # 명시적으로 타입을 맞춰줍니다.
            latent_action_tokens = latent_action_tokens.to(dtype=torch.bfloat16)
            visual_embed = visual_embed.to(dtype=torch.bfloat16)
            proprio_bfloat16 = proprio.to(dtype=torch.bfloat16)
            action_hist_bfloat16 = action_hist.to(dtype=torch.bfloat16)

            latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]
            hist_action = ''
            for latent_action_ids in generated_ids[0]:
                hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
            self.prev_hist_action.append(hist_action)
            
            if proprio_bfloat16.dim() == 1:
                proprio_bfloat16 = proprio_bfloat16.unsqueeze(0)
            
            predicted_actions_flat = self.action_decoder(
                latent_action_tokens, 
                visual_embed, 
                proprio_bfloat16,
                action_hist_bfloat16
            )
            
            # ActionDecoder의 출력 차원을 사용하여 reshape 합니다.
            action_dim_from_decoder = self.action_decoder.proj[0].out_features // self.pred_action_horizon
            predicted_actions = predicted_actions_flat.reshape(-1, self.pred_action_horizon, action_dim_from_decoder)
        
        return predicted_actions