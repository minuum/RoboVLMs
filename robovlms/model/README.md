
### 1.1. 최상위 파일

*   **`__init__.py`**: `model` 패키지를 Python 모듈로 인식하게 합니다. 현재 내용은 비어 있어, 외부에서 각 서브모듈을 직접 임포트해야 합니다.
*   **`vlm_builder.py`**:
    *   `build_vlm(vlm_config, tokenizer_config, precision)` 함수를 제공합니다.
    *   주어진 설정(`vlm_config`, `tokenizer_config`)에 따라 다양한 Vision-Language Model (PaliGemma, LLaVA 등)과 그에 맞는 Tokenizer(또는 Processor)를 Hugging Face Transformers 라이브러리를 사용하여 로드하고 반환합니다.
    *   `PaliGemma` 로드 시 `torch_dtype=torch.float32`로 설정하여 MPS 호환성을 확보하고, `device_map="cpu"`로 초기 로드 후 사용자가 `.to(device)`로 옮기도록 유도합니다.
*   **`flamingo_builder.py`**:
    *   OpenFlamingo 계열 모델을 빌드하기 위한 설정과 `build_llm_flamingo(llm_config)` 함수를 포함합니다.
    *   `RoboVLMs` 프로젝트가 Flamingo 모델도 지원했거나 지원할 계획이 있었음을 시사합니다. `vlm_builder.py`의 일반적인 VLM 로드 로직과 통합되거나 별도로 사용될 수 있습니다.

### 1.2. `action_encoder/`

*   **`linear_encoder.py`**: `ActionLinearEncoder` 클래스를 정의하며, 이는 이전 스텝의 Action을 다음 스텝의 입력으로 사용하고자 할 때, Action 벡터를 특정 차원의 임베딩으로 변환하는 간단한 Linear 레이어입니다. (예: 모방 학습이나 강화 학습에서 활용)

### 1.3. `backbone/`

*   **`__init__.py`**: 다양한 `RoboVLM` 구현체들(예: `RoboPaliGemma`, `RoboLLaVA`)을 쉽게 임포트할 수 있도록 `from .robopaligemma import RoboPaliGemma` 와 같은 구문들을 포함할 수 있습니다 (현재는 각 모델 파일명을 직접 지정해야 함).
*   **`base_backbone.py`**:
    *   VLA 모델의 핵심 추상 클래스인 **`BaseRoboVLM(nn.Module)`**을 정의합니다.
    *   `__init__`에서 설정(`configs`, `act_head_configs` 등)을 받아 VLM(`self.backbone`, `self.tokenizer`)과 Action Head(`self.act_head`) 등을 초기화합니다.
        *   VLM은 `vlm_builder.build_vlm`을 통해 로드됩니다.
        *   Action Head는 `act_head_configs`와 `policy_head` 모듈을 사용하여 동적으로 로드됩니다 (예: `getattr(action_heads_module, "MLPHead")`).
    *   `forward(vision_x, lang_x, ...)`: 모델의 메인 순전파 로직을 정의합니다. 입력된 시각 정보(`vision_x`)와 언어 정보(`lang_x`)를 받아, VLM을 통과시켜 특징을 추출하고, 이 특징을 Action Head로 전달하여 최종 Action을 예측합니다.
    *   다양한 보조 함수들(`encode_images`, `prepare_inputs_labels_for_multimodal`, `_forward_action_head` 등)을 포함하여 멀티모달 입력 처리, 학습 시 손실 계산 등을 담당합니다.
*   **`robopaligemma.py` (및 `robollava.py` 등 다른 VLM 파일들):**
    *   `BaseRoboVLM`을 상속받아 특정 VLM (예: PaliGemma)에 특화된 구현을 제공합니다.
    *   주로 해당 VLM의 이미지 인코딩 방식 (`model_encode_images` 오버라이드)이나 입력 처리 방식의 미세한 차이를 반영합니다.
    *   예를 들어, `RoboPaliGemma`는 PaliGemma 모델을 `self.backbone`으로 사용하고, `BaseRoboVLM`의 공통 파이프라인을 따릅니다.

### 1.4. `policy_head/`

*   **`__init__.py`**: `base_policy.py`에 정의된 주요 Action Decoder 클래스들(예: `LSTMDecoder`, `FCDecoder`)을 임포트하여 외부에서 `policy_head.LSTMDecoder` 형태로 사용할 수 있게 합니다.
*   **`base_policy.py`**:
    *   **`BasePolicyHead(nn.Module)`**: 모든 Action Decoder의 부모 클래스. `action_dim`, `action_space` 등을 정의하고, `loss` 계산을 위한 기본 인터페이스를 제공합니다.
    *   **`MLPHead`**: VLM 특징을 받아 여러 Linear 레이어를 통해 직접 Action 벡터를 출력합니다. (연속적인 Action 공간)
    *   **`FCDecoder`**: `MLPTanhHead` (팔 제어)와 `MLPSigmoidHead` (그리퍼 제어)를 결합하여 로봇 팔 제어에 특화된 연속 Action을 생성합니다.
    *   **`LSTMDecoder`, `GPTDecoder`**: 순차적인 Action 생성을 위해 각각 LSTM, GPT 아키텍처를 사용합니다.
    *   **`DiscreteDecoder`**: 연속적인 Action 공간을 이산화하거나, 미리 정의된 이산적인 Action Set 중 하나를 선택하는 문제를 다룹니다. `ActionTokenizer`를 사용하여 Action과 토큰 ID를 매핑합니다. (분류 문제로 접근)
*   **`action_tokenizer.py`**:
    *   `ActionTokenizer` 클래스를 정의합니다.
    *   연속적인 Action 값 (예: 로봇 팔 관절 각도)을 이산적인 토큰 ID로 변환 (양자화, `tokenize_actions`)하거나, 예측된 토큰 ID를 다시 연속적인 Action 값으로 복원 (`decode_token_ids_to_actions`)하는 역할을 합니다.
    *   `DiscreteDecoder`와 함께 사용되어 이산적인 Action 공간을 처리합니다.
*   **`trajectory_gpt2.py`**: `transformers`의 `GPT2Model`을 기반으로 한 Trajectory 생성용 Decoder를 구현합니다. `GPTDecoder`에서 사용될 수 있습니다.

### 1.5. `text_encoder/` 및 `vision_encoder/`

*   이 디렉토리들은 VLM에 내장된 인코더 외에, 독립적으로 텍스트나 비전 특징을 추출하거나 변환해야 할 때 사용될 수 있는 모듈들을 포함합니다.
*   **`clip_text_encoder.py`**: CLIP의 텍스트 인코더를 사용하여 텍스트 특징을 추출합니다.
*   **`vision_resampler.py`**: `PerceiverResampler` 등을 사용하여 Vision Transformer 등에서 나온 많은 수의 비전 토큰들을 고정된 개수의 적은 토큰으로 리샘플링(정보 요약/압축)합니다. `BaseRoboVLM`에서 `use_vision_resampler` 옵션으로 사용 여부를 결정할 수 있습니다.
*   **`vision_transformer.py`**: 간단한 Vision Transformer 구현 예시입니다.

## 2. VLA 파이프라인 구조도

다음은 `RoboVLMs`의 `model` 아키텍처를 기반으로 한 일반적인 VLA 파이프라인입니다. (학습 및 추론 시)

```mermaid
graph LR
    subgraph Input
        A[카메라 이미지 (Vision)]
        B[텍스트 명령어/프롬프트 (Language)]
        C[ (Optional) 이전 스텝 Action / 상태]
    end

    subgraph Preprocessing & Encoding
        D[Vision Preprocessing & Embedding] -- VLM 내부 Vision Encoder --> E
        F[Language Tokenizing & Embedding] -- VLM 내부 Text Encoder --> G
        H[ (Optional) Action Encoding] -- action_encoder.ActionLinearEncoder --> I
    end

    subgraph Core VLM & Fusion
        E[Vision Features]
        G[Language Features]
        I[ (Optional) Encoded Action Features]
        J[vlm_builder.py / backbone.RoboXXXModel] -- 멀티모달 퓨전 (예: Cross-Attention) --> K[Fused Multimodal Representation]
        K -- BaseRoboVLM.forward() --> L[Contextual Features / VLM Output Hidden State]
    end

    subgraph Action Decoding & Output
        L -- policy_head.BasePolicyHead (예: MLPHead, LSTMDecoder, DiscreteDecoder) --> M[Predicted Action]
        M --> N[후처리/실행 가능한 Action]
    end

    subgraph Execution / Environment Interaction
        N -- 로봇 제어기 / 시스템 API --> O[실제 Action 실행 (로봇 움직임, 화면 표시 등)]
    end

    A --> D
    B --> F
    C --> H

    D --> J
    F --> J
    H -.-> J


    %% 스타일링
    style Input fill:#DCDCDC,stroke:#333,stroke-width:2px
    style Preprocessing fill:#DCDCDC,stroke:#333,stroke-width:2px
    style "Core VLM & Fusion" fill:#ADD8E6,stroke:#333,stroke-width:2px
    style "Action Decoding & Output" fill:#ADD8E6,stroke:#333,stroke-width:2px
    style "Execution / Environment Interaction" fill:#90EE90,stroke:#333,stroke-width:2px
```

**파이프라인 설명:**

1.  **Input (입력):**
    *   카메라로부터 시각 정보 (이미지/비디오 프레임)를 받습니다.
    *   사용자 또는 시스템으로부터 언어 정보 (명령어, 질문, 현재 과제 등)를 받습니다.
    *   (선택적) 이전 타임스텝에서 수행했던 Action이나 현재 로봇/시스템의 상태 정보를 받을 수 있습니다.

2.  **Preprocessing & Encoding (전처리 및 인코딩):**
    *   **Vision:** 이미지는 VLM 내부의 Vision Encoder (예: ViT)를 통해 전처리되고 숫자 벡터(특징)로 변환됩니다. (`vision_encoder`의 `VisionResampler` 등이 여기서 활용될 수 있음)
    *   **Language:** 텍스트는 Tokenizer에 의해 토큰 ID 시퀀스로 변환된 후, VLM 내부의 Text Encoder (예: MPT, LLaMA의 일부)를 통해 임베딩 벡터로 변환됩니다.
    *   **Action (Optional):** 만약 이전 Action 정보를 현재 입력으로 사용한다면, `action_encoder`를 통해 임베딩됩니다.

3.  **Core VLM & Fusion (핵심 VLM 및 융합):**
    *   `vlm_builder.py`를 통해 로드된 VLM (예: `PaliGemma`, `LLaVA`) 또는 이를 `BaseRoboVLM`으로 감싼 `RoboPaliGemma` 등이 핵심 역할을 합니다.
    *   인코딩된 시각 특징과 언어 특징 (그리고 선택적으로 Action 특징)은 VLM 내부에서 퓨전 메커니즘 (예: 트랜스포머의 Cross-Attention 레이어)을 통해 결합되어, 두 양상(modality)의 정보를 모두 이해하는 **멀티모달 표현 (Fused Multimodal Representation)**을 형성합니다.
    *   `BaseRoboVLM`의 `forward` 메소드는 이 전체 과정을 관장하며, 최종적으로 다음 단계(Action Decoding)에 사용될 문맥적인 특징 벡터 (Contextual Features 또는 VLM의 마지막 레이어 은닉 상태)를 출력합니다.

4.  **Action Decoding & Output (액션 디코딩 및 출력):**
    *   VLM으로부터 나온 문맥적 특징 벡터는 `policy_head` 모듈의 Action Decoder (예: `MLPHead`, `LSTMDecoder`, `DiscreteDecoder` 등 `BaseRoboVLM`의 `self.act_head`로 초기화된 것)로 전달됩니다.
    *   Action Decoder는 이 특징을 기반으로 현재 상황에 가장 적합하다고 판단되는 **Action을 예측**합니다.
        *   `MLPHead`: 연속적인 제어 값 벡터 (예: 로봇 팔 관절 각도).
        *   `DiscreteDecoder`: 이산적인 Action 카테고리 ID (예: "물체 잡기", "문 열기").
    *   예측된 Action은 필요에 따라 후처리 (예: `action_tokenizer.py`를 사용한 디코딩, 값 범위 정규화)를 거쳐 실제 실행 가능한 형태로 변환됩니다.

5.  **Execution / Environment Interaction (실행 및 환경 상호작용):**
    *   최종적으로 결정된 Action은 실제 로봇 제어기(예: ROS)나 시스템 API (예: 화면 출력 함수, 파일 저장 함수)를 통해 물리적 세계 또는 가상 환경에서 실행됩니다.
    *   이 실행 결과는 다시 다음 타임스텝의 Vision 입력으로 반영되어 VLA 루프가 지속됩니다.

## 3. 현재 프로젝트(`jetson_vla_test.py` 기준)와의 매핑

현재 저희가 만들고 있는 `jetson_vla_test.py`는 위 파이프라인을 다음과 같이 단순화하여 사용하고 있습니다:

*   **Input:** 카메라 이미지 (Vision) + 고정된 프롬프트 또는 사용자 지정 프롬프트 (Language).
*   **Preprocessing & Encoding + Core VLM & Fusion:** `vlm_builder.py`를 통해 `PaliGemma`와 `AutoProcessor`를 직접 사용. `PaliGemmaForConditionalGeneration.generate()`가 이 부분과 "Action Decoding"의 언어 생성 부분을 함께 처리.
*   **Action Decoding & Output:**
    *   명시적인 `policy_head`를 사용하지 않음.
    *   VLM이 생성한 **텍스트 설명 자체가 일종의 "언어적 Action"**으로 간주됨.
    *   `CameraAction` Enum과 `execute_action` Python 함수를 통해, VLM의 텍스트 출력 또는 사용자의 초기 `--task` 지정에 따라 **미리 정의된 Python 기능(Action)**을 선택하고 실행.
*   **Execution:** `print`문으로 터미널에 결과를 출력하거나, `cv2.imwrite`로 파일을 저장.

이 README가 `RoboVLMs`의 `model` 아키텍처를 이해하는 데 도움이 되기를 바랍니다.