# RoboVLMs_upstream (수정본 - 실제 사용)

## ✅ 이 폴더가 실제로 사용되는 코드입니다

Mobile VLA 프로젝트를 위해 수정된 RoboVLMs 코드입니다.

---

## 📁 폴더 구조

```
RoboVLMs_upstream/
├── main.py                 # 학습 진입점
├── robovlms/
│   ├── data/
│   │   ├── mobile_vla_h5_dataset.py  # [수정됨] Mobile VLA H5 데이터셋
│   │   └── data_utils.py             # 데이터 유틸리티
│   ├── model/
│   │   ├── backbone/
│   │   │   └── base_backbone.py      # [수정됨] action_token Xavier 초기화
│   │   └── policy_head/
│   │       ├── mobile_vla_policy.py  # Mobile VLA LSTM Decoder
│   │       └── hybrid_action_head.py # [신규] Hybrid Action Head
│   └── train/
│       └── mobile_vla_trainer.py     # Mobile VLA 트레이너
└── runs/                   # 학습 결과
```

---

## 🔧 주요 수정 내역

### 1. action_token 초기화 (2025-12-09)
**파일**: `robovlms/model/backbone/base_backbone.py`
```python
# 기존: torch.zeros(hidden_size)
# 수정: Xavier 초기화
std = (2.0 / (hidden_size + hidden_size)) ** 0.5
self.action_token = nn.Parameter(torch.randn(hidden_size) * std)
```

### 2. abs_action 옵션 (2025-12-09)
**파일**: `robovlms/data/mobile_vla_h5_dataset.py`
```python
# linear_y 절대값 학습 (방향 제거)
if self.abs_action:
    actions_tensor[:, 1] = torch.abs(actions_tensor[:, 1])
```

### 3. Hybrid Action Head (2025-12-09)
**파일**: `robovlms/model/policy_head/hybrid_action_head.py`
- 방향: Binary Classification
- 크기: Continuous Regression

---

## 🚀 학습 실행

```bash
# 기본 학습
python3 RoboVLMs_upstream/main.py Mobile_VLA/configs/CONFIG_NAME.json

# 백그라운드 실행
nohup python3 RoboVLMs_upstream/main.py CONFIG.json > logs/train.log 2>&1 &
```

---

## 📊 현재 실험

| 케이스 | Config | 상태 |
|:---|:---|:---:|
| abs_action | mobile_vla_kosmos2_abs_action_20251209.json | 🔄 진행 중 |
| OpenVLA style | mobile_vla_openvla_style_20251209.json | 📋 대기 |
| No chunking | mobile_vla_no_chunk_20251209.json | 📋 대기 |

---

## 📁 관련 폴더

- **원본 참조**: `/home/billy/25-1kp/vla/RoboVLMs/` (수정 금지)
- **Config 파일**: `/home/billy/25-1kp/vla/Mobile_VLA/configs/`
- **학습 로그**: `/home/billy/25-1kp/vla/logs/`
- **체크포인트**: `/home/billy/25-1kp/vla/runs/`

---

## 🔗 Git 정보
- **Branch**: main
- **Commit**: 45b165e
- **마지막 동기화**: 2024-11-19

---

작성일: 2025-12-09
