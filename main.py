import os
import sys
import argparse
import json
from pathlib import Path
import importlib
import copy
import functools
from re import L
from typing import Dict, Any
import datetime

# Ensure lightning module can be imported
try:
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
except ImportError:
    # Try alternative import path
    try:
        import lightning
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    except ImportError:
        print("Error: Cannot import lightning module. Please ensure it is installed.")
        print(f"Python path: {sys.path}")
        print(f"Python executable: {sys.executable}")
        sys.exit(1)
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything
import torch
import torch.distributed as dist

from robovlms.train.base_trainer import BaseTrainer
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.datamodule.gr_datamodule import GRDataModule
from robovlms.data.data_utils import preprocess_image
from robovlms.utils.setup_callback import SetupCallback


def get_date_str():
    return str(datetime.date.today())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def init_lr_monitor_callback():
    return LearningRateMonitor(logging_interval="step")


def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(" ", "_"),
        logdir=config["log_dir"],
        ckptdir=config["output_dir"],
        cfgdir=config["log_dir"],
        config=config,
    )


def init_trainer_config(configs):
    # TODO: currently for other strategy we directly use the default settings.
    trainer_config = copy.deepcopy(configs["trainer"])
    trainer_config["devices"] = configs.get("gpus", "auto")
    trainer_config["num_nodes"] = configs.get("num_nodes", 1)
    trainer_config["gradient_clip_val"] = configs.get("gradient_clip_val", 0.0)
    
    # accelerator와 strategy 설정
    trainer_config["accelerator"] = configs.get("accelerator", "auto")
    raw_strategy = configs.get("strategy", None)

    # MPS 사용 시 precision 설정 강제
    # configs["trainer"]는 parse_args와 update_configs를 통해 커맨드라인 인자 또는 설정 파일 값을 가질 수 있음
    # 여기서 accelerator가 mps이면 precision을 강제로 "32-true"로 설정
    current_precision_from_config = configs.get("trainer", {}).get("precision")
    if trainer_config["accelerator"] == "mps":
        if current_precision_from_config != "32-true" and current_precision_from_config is not None: # None일 때는 메시지 출력 안함
            print(f"INFO: MPS accelerator detected. Overriding precision from '{current_precision_from_config}' to '32-true' for compatibility.")
        trainer_config["precision"] = "32-true"
    elif current_precision_from_config is not None:
        trainer_config["precision"] = current_precision_from_config
    else:
        # 기본 precision 설정 (MPS가 아니고, 설정 파일에도 없을 경우)
        trainer_config["precision"] = "32-true" # 기본값을 32-true로 설정

    exp_name = configs.get("exp_name", "default_exp")
    if exp_name is None:
        exp_name = "default_experiment_name_fallback"

    if trainer_config["accelerator"] == "mps" and raw_strategy == "ddp":
        print("Warning: DDP strategy is not fully supported on MPS. Consider using 'auto' or other compatible strategies.")
        trainer_config["strategy"] = "auto"
    elif raw_strategy == "ddp":
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True)
    elif raw_strategy:
        trainer_config["strategy"] = raw_strategy

    # init loggers
    loggers = None
    log_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["log_dir"] = configs["log_root"] / log_dir
    if isinstance(trainer_config.get("logger"), list):
        loggers = []
        for logger in trainer_config.get("logger"):
            if logger == "tensorboard":
                loggers.append(
                    TensorBoardLogger(configs["log_dir"].as_posix(), name=exp_name)
                )
            elif logger == "csv":
                loggers.append(CSVLogger(configs["log_dir"].as_posix(), name=exp_name))
            else:
                raise NotImplementedError

    trainer_config["logger"] = loggers

    ckpt_dir = Path(os.path.join(get_date_str(), exp_name))
    configs["output_dir"] = configs["output_root"] / ckpt_dir

    configs["log_dir"].mkdir(parents=True, exist_ok=True)
    configs["output_dir"].mkdir(parents=True, exist_ok=True)
    configs["cache_root"].mkdir(parents=True, exist_ok=True)
    # os.system(f"sudo chmod 777 -R runs/")

    configs["log_dir"] = configs["log_dir"].as_posix()
    configs["output_dir"] = configs["output_dir"].as_posix()
    configs.pop("output_root")
    configs.pop("log_root")
    configs["cache_root"] = configs["cache_root"].as_posix()

    trainer_config["callbacks"] = [
        init_setup_callback(configs),
        init_lr_monitor_callback(),
        ModelCheckpoint(
            dirpath=configs["output_dir"],
            save_top_k=3,              # 최고 성능 3개만 저장 (디스크 공간 절약)
            every_n_epochs=1,          # 매 epoch마다 체크 (최고 성능 갱신 시 저장)
            monitor="val_loss",        # validation loss 기준
            mode="min",                # loss 최소화
            save_last=True,            # 마지막 epoch도 저장
            filename="epoch_{epoch:02d}-val_loss={val_loss:.3f}"
        ),
    ]

    return trainer_config


def experiment(variant):
    seed_everything(variant["seed"] + int(os.environ.get("RANK", 0)))
    model_load_path = variant.get("model_load_path", None)

    # 1. model_url을 기반으로 실제 사용할 모델 저장소 이름 및 경로 결정
    true_repo_name = "default_model_repo"  # 기본값
    if variant.get("model_url"):
        repo_name_from_url = variant["model_url"].split("/")[-1]
        if repo_name_from_url.endswith(".git"):
            true_repo_name = repo_name_from_url[:-4]
        else:
            true_repo_name = repo_name_from_url
    elif variant.get("model_path"):
        path_parts = Path(variant['model_path']).parts
        if len(path_parts) > 0 and path_parts[0] == ".vlms":
            if len(path_parts) > 1:
                if path_parts[1] == "VLMs" and len(path_parts) > 2:
                    true_repo_name = path_parts[2]
                else:
                    true_repo_name = path_parts[1]
        else: 
            true_repo_name = Path(variant['model_path']).name

    true_model_path = os.path.join(".vlms", true_repo_name)
    print(f"DEBUG: Determined true_model_path: {true_model_path}")

    # HuggingFace Hub URL인지 판단
    model_url = variant.get("model_url", "") or ""
    is_hf_hub = isinstance(model_url, str) and ("huggingface.co" in model_url)

    # 2. 경로 강제 업데이트는 로컬 모델만 대상으로 함 (HF Hub는 원격 ID 유지)
    if not is_hf_hub:
        original_model_path_in_variant = variant.get('model_path')
        variant['model_path'] = true_model_path
        variant['model_config'] = os.path.join(true_model_path, "config.json")
        print(f"DEBUG: variant['model_path'] updated from '{original_model_path_in_variant}' to '{variant['model_path']}'")

        if "tokenizer" in variant and isinstance(variant["tokenizer"], dict):
            original_tokenizer_path = variant['tokenizer'].get('pretrained_model_name_or_path')
            variant["tokenizer"]["pretrained_model_name_or_path"] = true_model_path
            print(f"DEBUG: tokenizer path updated from '{original_tokenizer_path}' to '{true_model_path}'")
        else:
            print(f"DEBUG: tokenizer config not found or not a dict in variant.")

        if "vlm" in variant and isinstance(variant["vlm"], dict):
            original_vlm_path = variant['vlm'].get('pretrained_model_name_or_path')
            variant["vlm"]["pretrained_model_name_or_path"] = true_model_path
            print(f"DEBUG: vlm path updated from '{original_vlm_path}' to '{true_model_path}'")
        else:
            print(f"DEBUG: vlm config not found or not a dict in variant.")

    # 3. 모델 다운로드 (이제 통일되고 업데이트된 variant['model_path'] 기준)
    # 3. 모델 다운로드/준비
    if not is_hf_hub:
        if not os.path.exists(variant['model_path']):
            if variant.get("model_url"):
                print(
                    f"VLM backbone not found at {variant['model_path']}. Cloning {variant.get('model', true_repo_name)} from {variant['model_url']} into {variant['model_path']}..."
                )
                os.makedirs(os.path.dirname(variant['model_path']), exist_ok=True)
                os.system(f"git clone {variant['model_url']} {variant['model_path']}")
            else:
                error_msg = f"Model not found at {variant['model_path']} and model_url is not provided. Cannot download."
                print(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
        else:
            print(f"DEBUG: Model already exists at {variant['model_path']}. Skipping download.")
    else:
        print("INFO: Using HuggingFace Hub repo. Will let transformers download weights automatically.")
    
    trainer_config = init_trainer_config(variant)
    trainer = Trainer(**trainer_config)
    variant["gpus"] = trainer.num_devices
    variant["train_setup"]["precision"] = variant["trainer"]["precision"]

    if variant["fwd_head"] is not None:
        variant["train_setup"]["predict_forward_hand"] = variant["fwd_head"].get(
            "pred_hand_image", False
        )
    
    # Trainer 타입 선택 (Mobile VLA 또는 Base)
    trainer_type = variant.get("trainer_type", "BaseTrainer")
    if trainer_type == "MobileVLATrainer":
        TrainerClass = MobileVLATrainer
    else:
        TrainerClass = BaseTrainer
    
    model = TrainerClass.from_checkpoint(
        model_load_path, variant.get("model_load_source", "torch"), variant
    )

    # MPS 호환성을 위해 모델 전체를 float32로 명시적 변환
    if trainer_config.get("accelerator") == "mps":
        print("INFO: MPS accelerator detected. Explicitly casting the entire model to float32.")
        model = model.float()

    # Kosmos의 경우 processor 사용, 다른 모델은 image_processor 사용
    if variant["model"] == "kosmos" and hasattr(model.model, "processor"):
        image_preprocess = model.model.processor
    else:
        image_preprocess = model.model.image_processor

    _kwargs = {
        "model": model,
        "datamodule": GRDataModule(
            variant["train_dataset"],
            variant["val_dataset"],
            variant["batch_size"],
            variant["num_workers"],
            tokenizer=model.model.tokenizer,
            tokenizer_config=variant["tokenizer"],
            fwd_pred_next_n=variant["fwd_pred_next_n"],
            window_size=variant["window_size"],
            image_size=variant["image_size"],
            image_fn=functools.partial(
                preprocess_image,
                image_processor=image_preprocess,
                model_type=variant["model"],
            ),
            discrete=(
                False
                if variant["act_head"] is None
                else variant["act_head"].get("action_space", "continuous") == "discrete"
            ),
            discrete_action=(
                False
                if variant["act_head"] is None
                else variant["act_head"].get("action_space", "continuous") == "discrete"
            ),
            use_mu_law=variant.get("use_mu_law", False),
            mu_val=variant.get("mu_val", 255),
            n_bin=(
                256
                if variant["act_head"] is None
                else variant["act_head"].get("n_bin", 256)
            ),
            min_action=(
                -1
                if variant["act_head"] is None
                else variant["act_head"].get("min_action", -1)
            ),
            max_action=(
                1
                if variant["act_head"] is None
                else variant["act_head"].get("max_action", 1)
            ),
            discrete_action_history=variant.get("discrete_action_history", False),
            act_step=variant.get("fwd_pred_next_n", 1),
            norm_action=variant.get("norm_action", False),
            norm_min=variant.get("norm_min", -1),
            norm_max=variant.get("norm_max", 1),
            regular_action=variant.get("regular_action", False),
            x_mean=variant.get("x_mean", 0),
            x_std=variant.get("x_std", 1),
            weights=variant.get("train_weights", None),
            tcp_rel=variant.get("tcp_rel", False),
            # vit_name=vit_name,
            model_name=variant.get("model", "flamingo"),
        ),
        "ckpt_path": variant["resume"],
    }
    if _kwargs["ckpt_path"] is not None:
        print(f"Resuming from {variant['resume']}...")

    trainer.fit(**_kwargs)


def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def update_configs(configs, args):
    configs["raw_config_path"] = args["config"]
    configs["output_root"] = (
        Path(configs["output_root"]) / configs["model"] / configs["task_name"]
    )
    configs["log_root"] = (
        Path(configs["log_root"]) / configs["model"] / configs["task_name"]
    )
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]

    for k, v in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_v != None:
                    configs[k][sub_k] = sub_v
        else:
            if v != None:
                configs[k] = v
    return configs


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("config", type=str, help="config file used for training")
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--num_nodes", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--annotation_file", default=None, type=str)
    parser.add_argument("--model_load_path", default=None, type=str)
    parser.add_argument("--data_subfolder", default=None, type=str)
    parser.add_argument("--task_num", default=None, type=int)
    parser.add_argument("--seq_len", default=None, type=float)
    parser.add_argument("--exp_name", default=None, type=str)

    # Loss
    parser.add_argument("--arm_gripper_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_loss_ratio", default=None, type=float)
    parser.add_argument("--fwd_pred_next_n", default=None, type=int)

    parser.add_argument("--use_multi_modal_emb", default=False, action="store_true")
    parser.add_argument(
        "--no_video_pretrained_model", default=False, action="store_true"
    )
    parser.add_argument("--finetune", default=False, action="store_true")

    # Training
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--min_lr_scale", default=None, type=float)
    parser.add_argument("--warmup_epochs", default=None, type=int)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)

    global_names = set(vars(parser.parse_known_args()[0]).keys())

    # Trainer
    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--strategy", default=None, type=str)
    trainer_parser.add_argument("--accelerator", default=None, type=str)
    trainer_parser.add_argument("--precision", default=None, type=str)
    trainer_parser.add_argument("--gradient_clip_val", default=None, type=float)
    trainer_parser.add_argument("--max_epochs", default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    # Model Architecture
    llm_parser = parser.add_argument_group("llm")
    llm_parser.add_argument("--type", default=None, type=str)
    llm_parser.add_argument("--n_embd", default=None, type=int)
    llm_parser.add_argument("--n_layer", default=None, type=int)
    llm_parser.add_argument("--n_head", default=None, type=int)
    llm_names = (
        set(vars(parser.parse_known_args()[0]).keys()) - global_names - trainer_names
    )

    args = {}
    trainer_args = {}
    llm_args = {}
    temp_args = vars(parser.parse_args())
    for k, v in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v
        elif k in llm_names:
            llm_args[k] = v

    args["llm"] = llm_args
    args["trainer"] = trainer_args

    return args


if __name__ == "__main__":
    # import os

    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    args = parse_args()

    # load config files
    configs = load_config(args.get("config"))
    configs = update_configs(configs, args)

    # 분산 처리 초기화 조건 명확히 수정
    is_ddp_strategy = False
    # trainer 그룹의 strategy 확인
    trainer_strategy_conf = configs.get("trainer", {}).get("strategy")
    if isinstance(trainer_strategy_conf, str) and "ddp" in trainer_strategy_conf.lower():
        is_ddp_strategy = True
    elif isinstance(trainer_strategy_conf, DDPStrategy):
        is_ddp_strategy = True
    
    # configs 루트 레벨의 strategy 확인 (커맨드 라인 인자 우선)
    config_strategy_conf = configs.get("strategy")
    if isinstance(config_strategy_conf, str) and "ddp" in config_strategy_conf.lower():
        is_ddp_strategy = True

    if configs.get("accelerator") != "mps" and is_ddp_strategy:
        if dist.is_available() and not dist.is_initialized():
            print("Initializing process group for DDP...")
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            if os.environ.get("RANK") is None or os.environ.get("WORLD_SIZE") is None:
                 print("Warning: RANK and WORLD_SIZE environment variables are not set for DDP. This might lead to errors if not using torchrun.")
            dist.init_process_group(backend=backend)
    elif configs.get("accelerator") == "mps":
        print("MPS accelerator selected. Skipping explicit process group initialization.")
    
    experiment(variant=configs)
