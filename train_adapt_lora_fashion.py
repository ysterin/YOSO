#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import glob
import logging
import os
import shutil
from pathlib import Path
import accelerate
# import datasets
import numpy as np
import transformers
from PIL import Image, ImageOps
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from safetensors import safe_open
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from unet import UNet2DConditionModel
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, DDIMScheduler, EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import torch
from torch import nn
import math

from utils import normalization, AttentionPool2d, compute_snr, compute_snr_sqrt, append_dims, \
    get_module_kohya_state_dict

# if is_wandb_available():
#     pass

import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

def sample_from_model(model, noise_scheduler, n_steps, alphas, sigmas, noise, prediction_type, encoder_hidden_states):
    sample = noise
    timesteps = np.linspace(len(noise_scheduler.timesteps) - 1, 0, n_steps + 1, dtype=np.int32)[:-1]
    for i, timestep in enumerate(timesteps):
        t = torch.LongTensor(np.array([timestep]))
        t = t.to(sample.device)
        if i > 0:
            noisy_sample = noise_scheduler.add_noise(sample, noise, t)
        else:
            noisy_sample = sample
        model_output = model(noisy_sample, t, encoder_hidden_states, return_dict=False)[0]
        sample = predicted_origin(model_output, t, noisy_sample, prediction_type, alphas, sigmas)
        # c_skip_start, c_out_start = scalings_for_boundary_conditions(t)
        # c_skip_start, c_out_start = [append_dims(x, sample.ndim) for x in [c_skip_start, c_out_start]]
        # c_skip_start, c_out_start = [x.to(sample.dtype) for x in [c_skip_start, c_out_start]]
        # sample = c_skip_start * noisy_sample + c_out_start * sample

    return sample


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data ** 2 / ((timestep / 0.1) ** 2 + sigma_data ** 2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="yoso_lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lambda_con",
        type=float,
        default=2,
        help="The weight for consistency loss."
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=1.0,
        help="The weight for KL loss",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--start_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be started from a specific checkpoint. Use a path saved by"
            ' `--checkpointing_steps`.'
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "-w",
        "--wandb",
        type=str,
        default=None,
        help="api",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="Whether to zero the terminal SNR.",
    )
    parser.add_argument(
        "--adapt_zero_terminal_snr",
        action="store_true",
        help="Whether to use adaptation loss for the terminal SNR.",
    )
    parser.add_argument(
        "--adapt_v_prediction",
        action="store_true",
        help="Whether to use adaptation loss for the v prediction.",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Whether to use Lora or not."
    )



    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def main():
    args = parse_args()

    data_path = os.getenv("DATA_PATH", "/mnt/data")
    artifacts_path = os.getenv("ARTIFACTS_PATH", "/mnt/artifacts")

    if args.output_dir.startswith("/artifacts/"):
        args.output_dir = os.path.join(artifacts_path, "/".join(args.output_dir.split("/")[2:]))

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    import torch

    if args.wandb is not None:
        os.environ['WANDB_API_KEY'] = os.environ.get('WANDB_API_KEY', args.wandb)
        wandb.login(key=args.wandb)

    if accelerator.is_main_process:
        wandb.init(project=args.tracker_project_name, config=args)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    timestep_spacing="trailing",
                                                    rescale_betas_zero_snr=args.zero_terminal_snr,
                                                    prediction_type=args.prediction_type)

    teacher_noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    timestep_spacing="trailing",
                                                    rescale_betas_zero_snr=args.zero_terminal_snr and not args.adapt_zero_terminal_snr,
                                                    prediction_type=args.prediction_type and not args.adapt_v_prediction)

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant,
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    if args.adapt_zero_terminal_snr or args.adapt_v_prediction:
        teacher_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
    from diffusers.models import ModelMixin
    from diffusers.configuration_utils import ConfigMixin
    class My_Dis(ModelMixin, ConfigMixin, ):
        def __init__(self):
            super(My_Dis, self).__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="unet", revision=args.revision, variant=args.variant
            )
            self.out_head = nn.ModuleList([
                normalization(1280),
                nn.SiLU(),
                AttentionPool2d(
                    (args.resolution // 64), 1280, 16, 512),
                nn.SiLU(),
                nn.Linear(512, 1),
            ])
            self.unet.up_blocks = None
            self.unet.conv_norm_out = None
            self.unet.conv_act = None
            self.unet.conv_out = None  # Remove the un-used part

        def forward(self, latents, timesteps, encoder_hidden_states, return_dict=False):
            features = self.unet(latents, timesteps, encoder_hidden_states, return_half=True)
            for layer in self.out_head:
                features = layer(features)
            output = features
            return output

    # unet_gan = My_Dis()
    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    # unet_gan.train()
    if not args.no_lora:
        lora_config = LoraConfig(
            r=64,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ],
        )
        unet = get_peft_model(unet, lora_config)
        if args.start_from_checkpoint is not None:
            if args.start_from_checkpoint.startswith("/artifacts/"):
                args.start_from_checkpoint = os.path.join(artifacts_path, "/".join(args.start_from_checkpoint.split("/")[2:]))

            state_dict = {}
            with safe_open(args.start_from_checkpoint, framework="pt", device=0) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)

            def adapt_state_dict(sd):
                new_state_dict = {}
                for k, v in sd.items():
                    new_k = ".".join(k.split(".")[:-1] + ['default'] + k.split(".")[-1:])
                    new_state_dict[new_k] = v
                return new_state_dict

            state_dict = adapt_state_dict(state_dict)

            # Load only the LoRA weights
            missing_keys, unexpected_keys = unet.load_state_dict(
                state_dict,
                strict=False
            )
            # unet.load_adapter(args.start_from_checkpoint, "default", is_trainable=True)
    # ema_lora_state_dict =  get_peft_model_state_dict(unet_, adapter_name="default")
    # unet_gan.unet = get_peft_model(unet_gan.unet, lora_config)
    from copy import deepcopy
    # Create EMA for the unet.
    if args.use_ema and not args.no_lora:
        # dic_lora = get_peft_model_state_dict(unet, adapter_name="default")
        dic_lora = get_module_kohya_state_dict(unet, "lora_unet", torch.float32)
        ema_dic_lora = deepcopy(dic_lora)
        # ema_unet = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        # )
        # ema_unet = deepcopy(unet)
        # ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            # unet.enable_xformers_memory_efficient_attention()
            # unet_gan.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                if args.no_lora:
                    unet_.save_pretrained(os.path.join(output_dir, "unet"))
                    if args.use_ema:
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                else:
                    lora_state_dict = get_peft_model_state_dict(unet_, adapter_name="default")
                    StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "unet_lora"), lora_state_dict)
                    unet_.save_pretrained(os.path.join(output_dir, "unet"))
                    if args.use_ema:
                        torch.save(ema_dic_lora, os.path.join(output_dir, 'ema_lora.pt'))
                # ema_dic_lora
                # ema_lora_state_dict = get_peft_model_state_dict(unet_, adapter_name="default")
                # StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "unet_lora_ema"), ema_lora_state_dict)
                # ema_unet.save_pretrained(os.path.join(output_dir, "unet"))

                for i, model in enumerate(models):
                    weights.pop()

        def load_model_hook(models, input_dir):
            # if args.use_ema:
            #     load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            #     ema_unet.load_state_dict(load_model.state_dict())
            #     ema_unet.to(accelerator.device)
            #     del load_model
            unet_ = accelerator.unwrap_model(unet)
            unet_.load_adapter(os.path.join(input_dir, "unet"), "default", is_trainable=True)

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
            #     if isinstance(model, My_Dis):
            #         # save_pth_ = os.path.join(output_dir, "unetGAN")
            #         # checkpoint = torch.load(os.path.join('unet_gan.pth'))
            #         # model.load_state_dict(checkpoint)
            #         load_model = My_Dis.from_pretrained(input_dir, subfolder="unetGAN")
            #         model.register_to_config(**load_model.config)
            #         model.load_state_dict(load_model.state_dict())
            #         del load_model
            #     else:
            #         # load diffusers style into model
            #         load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            #         model.register_to_config(**load_model.config)

            #         model.load_state_dict(load_model.state_dict())
            #         del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # unet_gan.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    import torch
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer_d = optimizer_cls(
    #     unet_gan.parameters(),
    #     lr=args.learning_rate,
    #     betas=(0., args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    import json
    import torch
    from torch.utils.data import Dataset

    # class CustomImagePromptDataset(Dataset):
    #     def __init__(self, jsonl_file, transform=None):
    #         self.data = []
    #         self.transform = transform
    #         self.tokenizer = CLIPTokenizer.from_pretrained(
    #             'runwayml/stable-diffusion-v1-5', subfolder="tokenizer", )
    #         with open(jsonl_file, 'r') as file:
    #             for line in file:
    #                 entry = json.loads(line)
    #                 self.data.append(entry['prompt'])
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, idx):
    #         text = self.data[idx]
    #         prompt = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length",
    #                                 truncation=True, return_tensors="pt").input_ids
    #         return text, prompt

    class CustomImagePromptDataset(Dataset):
        def __init__(self, root_dir, frames_dirname="frames", annots_file_name="txts_flash_yoso.json", transform=None):
            # self.data = []
            self.transform = transform
            self.tokenizer = CLIPTokenizer.from_pretrained(
                'runwayml/stable-diffusion-v1-5', subfolder="tokenizer", )
            self.root_dir = root_dir
            # self.subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and
            #                 os.path.isfile(os.path.join(root_dir, d, annots_file_name))]
            self.annots_files = glob.glob(os.path.join(root_dir, "*", annots_file_name))
            self.file_paths = []
            self.annots_dict = {}
            for annots_file in self.annots_files:
                folder_name = os.path.basename(os.path.dirname(annots_file))
                with open(annots_file, 'r') as file:
                    annots = json.load(file)
                    for k, v in annots.items():
                        file_path = os.path.join(root_dir, folder_name, frames_dirname, k)
                        annot = v[0]["11"]
                        if not isinstance(annot, str):
                            continue
                        self.annots_dict[file_path] = annot
                        self.file_paths.append(file_path)

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            file_path = self.file_paths[idx]
            text = self.annots_dict[file_path]
            prompt = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors="pt").input_ids
            image = Image.open(file_path).convert('RGB')
            image = ImageOps.pad(image, (args.resolution, args.resolution))
            if self.transform:
                image = self.transform(image)

            return text, prompt, image


    # Create Dataset
    # dataset = CustomImagePromptDataset(jsonl_file=os.path.join(data_path, 'YOSO/train_anno.jsonl'), transform=None)
    # dataset = CustomImagePromptDataset(jsonl_file='./train_anno.jsonl', transform=None)
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # dataset = CustomImagePromptDataset(root_dir=f"{data_path}/fashion/fashion-data-control/retrival_frames/frames",
    #                                    transform=transform)

    dataset = CustomImagePromptDataset(root_dir=f"{data_path}/fashion/feb/shops/fashion-shops-plus-size-unfiltered",
                                       transform=transform)


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=8,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # unet_gan.unet = accelerator.prepare_model(unet_gan.unet,find_unused_parameters = True)
    # unet_gan.out_head = accelerator.prepare_model(unet_gan.out_head)
    # unet_gan = accelerator.prepare_model(unet_gan)

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler  # , find_unused_parameters=True
    )
    # if args.use_ema:
    #     ema_unet.to(accelerator.device)

    from copy import deepcopy
    # Create EMA for the unet.
    if args.use_ema and args.no_lora:
        # dic_lora = get_peft_model_state_dict(unet, adapter_name="default")
        # dic_lora = get_module_kohya_state_dict(unet, "lora_unet", torch.float32)
        # ema_dic_lora = deepcopy(dic_lora)
        # ema_unet = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        # )
        # ema_unet = deepcopy(unet)
        if isinstance(unet, DistributedDataParallel):
            ema_unet = deepcopy(unet.module)
        else:
            ema_unet = deepcopy(unet)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config,
                            decay=0.999)
        # ema_unet = accelerator.prepare_model(ema_unet)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # unet_sd = UNet2DConditionModel.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", subfolder="unet", revision=args.revision, variant=args.variant)
    # unet_sd.enable_xformers_memory_efficient_attention()
    # unet_sd.up_blocks = None
    # unet_sd.conv_norm_out = None
    # unet_sd.conv_act = None
    # unet_sd.conv_out = None  # Remove the un-used part
    # unet_sd = accelerator.prepare_model(unet_sd)
    # unet_sd.requires_grad_(False)
    if args.adapt_zero_terminal_snr or args.adapt_v_prediction:
        teacher_unet = accelerator.prepare_model(teacher_unet)
        teacher_unet.requires_grad_(False)
        teacher_unet.eval()

    # Potentially load in the weights and states from a previous save
    print(args.resume_from_checkpoint)
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.startswith("/artifacts/"):
            args.resume_from_checkpoint = os.path.join(artifacts_path, "/".join(args.resume_from_checkpoint.split("/")[2:]))
            path = args.resume_from_checkpoint
        elif args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        elif path.startswith(artifacts_path):
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = 0
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    from copy import deepcopy
    import torch.nn.functional as F
    # unet_sd.eval()
    # unet_sd.requires_grad_(False)
    from diffusers import AutoPipelineForText2Image

    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    # pipe_sdxl = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,  variant="fp16")
    # pipe_sdxl.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    # pipe_sdxl = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", vae=vae, torch_dtype=torch.float16,
    #                                                       variant="fp16")
    # pipe_sdxl.to(accelerator.device)
    # pipe_sdxl.set_progress_bar_config(disable=True)
    # pipe_sdxl.enable_xformers_memory_efficient_attention()
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="pipeline", revision=args.revision, torch_dtype=weight_dtype
    )
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.scheduler = deepcopy(noise_scheduler)
    pipeline.unet = unet

    # pipe_sdxl.to(unet.device)
    # pipe_sdxl.unet.eval()
    # pipe_sdxl.unet.requires_grad_(False)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space\
                text_ = list(batch[0])
                # noise = torch.randn([len(text_), 4, 64, 64])
                # with torch.no_grad():
                #     # Generate Data By SD-turbo with 2 steps, instead of use JourneyDB which yeilds the style shift.
                #     latents = pipe_sdxl(prompt=text_, num_inference_steps=1, guidance_scale=0.0, output_type="latent",
                #                         latents=noise.to(pipe_sdxl.unet.dtype))[0]
                images = batch[2].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor  # get latent of real data.
                #     latents = pipe_sdxl.vae.encode(images).latent_dist.sample()

                #     images = pipe_sdxl.vae.decode(latents / pipe_sdxl.vae.config.scaling_factor, return_dict=False)[0].to(weight_dtype).clamp(-1,1)
                # latents = vae.encode(images).latent_dist.sample()
                # latents = latents * vae.config.scaling_factor # get latent of real data.
                # Do not need encode. As the SD-2.1 share the same encoder with SD-1.5


                # Sample noise that we'll add to the latents
                # noise = torch.randn_like(latents)
                noise = torch.randn(latents.shape, device=latents.device, dtype=weight_dtype)
                # args.noise_offset = 0.05
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                input_ids = batch[1]  # .to(weight_dtype)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                c_skip_start, c_out_start = scalings_for_boundary_conditions(timesteps)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip_start, c_out_start = [x.to(latents.dtype) for x in [c_skip_start, c_out_start]]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                pred_x_0 = predicted_origin(
                    model_pred,
                    timesteps,
                    noisy_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                pred_x_0 = c_skip_start * noisy_latents + c_out_start * pred_x_0

                args.snr_gamma = 5
                snr = compute_snr(noise_scheduler, timesteps)
                real_mse_loss_weights = \
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                kl_loss = F.mse_loss(pred_x_0.float(), latents.float(), reduction='none')
                kl_loss = kl_loss.mean(dim=list(range(1, len(kl_loss.shape))))
                kl_loss_unweighted = kl_loss.mean()
                kl_loss = kl_loss * real_mse_loss_weights

                kl_loss = kl_loss.mean()
                avg_kl_loss = accelerator.gather(kl_loss.repeat(args.train_batch_size)).mean()

                # # Backpropagate
                # accelerator.backward(args.lambda_kl * kl_loss)
                log_dict = {"kl_loss": avg_kl_loss / args.gradient_accumulation_steps,
                               "total_loss": args.lambda_kl * kl_loss}
                if args.adapt_zero_terminal_snr:
                    teacher_noisy_latents = teacher_noise_scheduler.add_noise(latents, noise, timesteps)
                    teacher_model_pred = teacher_unet(teacher_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                    adapt_zero_terminal_snr_loss = F.mse_loss(model_pred.float(), teacher_model_pred.float(), reduction='none')
                    adapt_zero_terminal_snr_loss = adapt_zero_terminal_snr_loss.mean(dim=list(range(1, len(adapt_zero_terminal_snr_loss.shape))))
                    adapt_zero_terminal_snr_loss = adapt_zero_terminal_snr_loss.mean()
                    avg_adapt_zero_terminal_snr_loss = accelerator.gather(adapt_zero_terminal_snr_loss.repeat(args.train_batch_size)).mean()
                    log_dict["adapt_zero_terminal_snr_loss"] = avg_adapt_zero_terminal_snr_loss / args.gradient_accumulation_steps
                    # Backpropagate
                    accelerator.backward(args.lambda_kl * adapt_zero_terminal_snr_loss)
                elif args.prediction_type == "v_prediction":
                    v = noise_scheduler.get_velocity(latents, noise, timesteps)
                    v_loss = F.mse_loss(v.float(), model_pred.float(), reduction='none')
                    v_loss = v_loss.mean(dim=list(range(1, len(v_loss.shape))))
                    v_loss = v_loss.mean()
                    kl_loss = v_loss
                    avg_v_loss = accelerator.gather(v_loss.repeat(args.train_batch_size)).mean()
                    log_dict["v_loss"] = avg_v_loss / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(args.lambda_kl * v_loss)
                else:
                    # Backpropagate
                    accelerator.backward(args.lambda_kl * kl_loss)

                if accelerator.is_main_process:
                    # Log the losses
                    wandb.log(log_dict, step=global_step)
                    # wandb.log({"kl_loss": avg_kl_loss / args.gradient_accumulation_steps,
                    #            "total_loss": args.lambda_kl * kl_loss}, step=global_step)
                    # if args.prediction_type == "v_prediction":
                    #     wandb.log({"v_loss": avg_v_loss / args.gradient_accumulation_steps}, step=global_step)


                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if step % 100 == 0:
                    with torch.no_grad():
                        save_dir = f'./{args.output_dir}/images/{step}'
                        os.makedirs(save_dir, exist_ok=True)
                        T_ = torch.randint(noise_scheduler.config.num_train_timesteps - 1,
                                           noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        T_ = T_.long()
                        images_t = vae.decode(pred_x_0.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
                        # images_pervt = vae.decode(target.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[
                        #     0]
                        # pure_noisy = noise_scheduler.add_noise(latents, noise, T_)
                        pure_noisy = torch.randn_like(latents)
                        noise_pred = unet(pure_noisy, T_, encoder_hidden_states, return_dict=False)[0]
                        noise_generation = predicted_origin(
                            noise_pred,
                            T_,
                            pure_noisy,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        n_steps = 25
                        mutlistep_sample = sample_from_model(
                            model=unet,
                            noise_scheduler=noise_scheduler,
                            n_steps=n_steps,
                            noise=pure_noisy,
                            alphas=alpha_schedule,
                            sigmas=sigma_schedule,
                            prediction_type=noise_scheduler.config.prediction_type,
                            encoder_hidden_states=encoder_hidden_states
                        )
                        with torch.autocast(device_type="cuda", dtype=weight_dtype):
                            images_pipeline = pipeline(prompt=text_, num_inference_steps=n_steps, guidance_scale=1.0,
                                                       output_type="pt", return_dict=False, latents=pure_noisy)[0]
                            images_pipeline_cfg = pipeline(prompt=text_, num_inference_steps=n_steps, guidance_scale=3.0,
                                                       output_type="pt", return_dict=False, latents=pure_noisy)[0]

                        images_noise = \
                            vae.decode(noise_generation.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
                        images_real = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[
                            0]
                        images_multistep = \
                            vae.decode(mutlistep_sample.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
                    if accelerator.is_main_process:
                        images_t = images_t.clamp(-1, 1) * 0.5 + 0.5
                        # images_pervt = images_pervt.clamp(-1, 1) * 0.5 + 0.5
                        images_noise = images_noise.clamp(-1, 1) * 0.5 + 0.5
                        images_multistep = images_multistep.clamp(-1, 1) * 0.5 + 0.5
                        images_real = images_real.clamp(-1, 1) * 0.5 + 0.5
                        images_pipeline = images_pipeline.clamp(-1, 1) * 0.5 + 0.5
                        images_pipeline_cfg = images_pipeline_cfg.clamp(-1, 1) * 0.5 + 0.5
                        # save_image(images_t, f'./{args.output_dir}/iamges_t_selfper.jpg', normalize=False, nrow=4)
                        # save_image(images_pervt, f'./{args.output_dir}/images_prevt_selfper.jpg', normalize=False,
                        #            nrow=4)
                        save_image(images_t, os.path.join(save_dir, 'iamges_t_selfper.jpg'), normalize=False, nrow=4)
                        # save_image(images_pervt, os.path.join(save_dir, 'images_prevt_selfper.jpg'), normalize=False,
                        #            nrow=4)
                        save_image(images_noise, os.path.join(save_dir, 'singlestep.jpg'), normalize=False, nrow=4)
                        # save_image(images_real.clamp(-1, 1) * 0.5 + 0.5, f'./{args.output_dir}/real_data.jpg',
                        #            normalize=False, nrow=4)
                        save_image(images_real.clamp(-1, 1) * 0.5 + 0.5, os.path.join(save_dir, 'real_data.jpg'),
                                   normalize=False, nrow=4)
                        # save_image(images_multistep, os.path.join(save_dir, 'multistep.jpg'), normalize=False, nrow=4)
                        save_image(images_pipeline, os.path.join(save_dir, 'pipeline.jpg'), normalize=False, nrow=4)
                        save_image(images_pipeline_cfg, os.path.join(save_dir, 'pipeline_cfg.jpg'), normalize=False, nrow=4)

                        # save_image(images_multistep, f'./{args.output_dir}/multistep.jpg', normalize=False, nrow=4)
                        wandb.log({"images_t": wandb.Image(images_t),
                                   "images_singlestep": wandb.Image(images_noise),
                                      "images_multistep": wandb.Image(images_multistep),
                                   "images_pipeline": wandb.Image(images_pipeline),
                                   "images_pipeline_cfg": wandb.Image(images_pipeline_cfg),
                                   "images_real": wandb.Image(images_real)}, step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    decay = 0.999
                    unet_ = accelerator.unwrap_model(unet)
                    dic_lora = get_module_kohya_state_dict(unet_, "lora_unet", torch.float32)
                    for k, v in dic_lora.items():
                        ema_dic_lora[k] = decay * deepcopy(ema_dic_lora[k]).to(v.device) + (1 - decay) * (deepcopy(v))
                    # ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                if accelerator.is_main_process:
                    wandb.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": kl_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    accelerator.end_training()


if __name__ == "__main__":
    main()
