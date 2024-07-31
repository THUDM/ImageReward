'''
@File       :   ReFL_SDXL.py
@Description:   ReFL Algorithm using StableDiffusionXLPipeline.
* Based on diffusers code base
* https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py
'''

"""Fine-tuning script for Stable Diffusion XL for text2image."""

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import ImageReward as RM

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0")

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {}

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, cache_dir: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder=subfolder, 
        revision=revision,
        cache_dir=cache_dir,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--grad_scale", type=float, default=1e-3, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
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
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
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
        default=1024,
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
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam self.optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam self.optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam self.optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `self.noise_scheduler.config.prediction_type` is chosen.",
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # Personal usage arguments
    parser.add_argument(
        "--apply_pre_loss", action="store_true", help="Whether or not to apply pretrained loss."
    )
    parser.add_argument(
        "--apply_reward_loss", action="store_true", help="Whether or not to apply reward loss."
    )
    parser.add_argument(
        "--image_reward_version",
        type=str,
        required=True,
        help=(
            "The version of ImageReward to load with. This could be a downloadable version, i.e. ImageReward-v1.0"
            "or a file path."),
    )
    parser.add_argument(
        "--save_only_one_ckpt",
        action="store_true",
        help="If given, then it only stores one checkpoint through the whole training, the one with the best training loss."
        "This is useful to manage the storage."
    )
    parser.add_argument(
        "--image_base_dir",
        default="",
        help="The base directory where stored all the images."
    )
    parser.add_argument(
        "--mapping_batch_size",
        default=128,
        type=int,
        help="Batch size to map huggingface data."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def encode_prompt_rm(batch, caption_column, reward_model):
    prompt_batch = batch[caption_column]

    text_inputs = reward_model.blip.tokenizer(prompt_batch, padding='max_length', truncation=True, max_length=35, return_tensors="pt")

    return {
        "rm_input_ids": text_inputs.input_ids,
        "rm_attention_mask": text_inputs.attention_mask,
    }

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt    
def encode_prompt(batch, text_encoders, tokenizers, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return {"model_input": model_input.cpu()}


class Trainer(object):
    def __init__(self, pretrained_model_name_or_path, train_data_dir, args) -> None:

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        logging_dir = os.path.join(args.output_dir, args.logging_dir)

        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            project_dir=logging_dir
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if args.output_dir is not None and not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)

        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            cache_dir=args.cache_dir
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            use_fast=False,
            cache_dir=args.cache_dir
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, args.revision, args.cache_dir
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, args.revision, args.cache_dir, subfolder="text_encoder_2"
        )

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, 
                                                             subfolder="scheduler",
                                                             cache_dir=args.cache_dir
        )

        # Check for terminal SNR in combination with SNR Gamma
        text_encoder_one = text_encoder_cls_one.from_pretrained(self.pretrained_model_name_or_path, 
                                                                subfolder="text_encoder", 
                                                                revision=args.revision,
                                                                cache_dir=args.cache_dir
        )

        text_encoder_two = text_encoder_cls_two.from_pretrained(self.pretrained_model_name_or_path, 
                                                                subfolder="text_encoder_2", 
                                                                revision=args.revision,
                                                                cache_dir=args.cache_dir
        )
            
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            cache_dir=args.cache_dir
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="unet", 
            revision=args.revision,
            cache_dir=args.cache_dir
        )

        self.reward_model = RM.load("ImageReward-v1.0", device=self.accelerator.device)


        # Freeze vae and text encoders.
        self.vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        self.reward_model.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move self.unet, vae, reward model, and text_encoder to device and cast to self.weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)
        self.reward_model.to(self.accelerator.device, dtype=self.weight_dtype)

        # Create EMA for the self.unet.
        if args.use_ema:
            self.ema_unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_name_or_path, 
                subfolder="unet", 
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
            self.ema_unet = EMAModel(self.ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.ema_unet.config)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `self.accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_main_process:
                    if args.use_ema:
                        self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                        # make sure to pop weight so that corresponding model is not saved again
                        if weights:
                            weights.pop()

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    self.ema_unet.load_state_dict(load_model.state_dict())
                    self.ema_unet.to(self.accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, 
                                                                      subfolder="unet",
                                                                      cache_dir=args.cache_dir
                    )
                    
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * self.accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = self.unet.parameters()
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
        else:
            data_files = {}
            data_files["train"] = train_data_dir
            dataset = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        def preprocess_train(examples):
            images = [Image.open(os.path.join(args.image_base_dir, im_file)).convert("RGB") for im_file in examples[image_column]]

            # image aug
            original_sizes = []
            all_images = []
            crop_top_lefts = []
            for image in images:
                original_sizes.append((image.height, image.width))
                image = train_resize(image)
                if args.random_flip and random.random() < 0.5:
                    # flip
                    image = train_flip(image)
                if args.center_crop:
                    y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                    x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                    image = crop(image, y1, x1, h, w)
                crop_top_left = (y1, x1)
                crop_top_lefts.append(crop_top_left)
                image = train_transforms(image)
                all_images.append(image)

            examples["original_sizes"] = original_sizes
            examples["crop_top_lefts"] = crop_top_lefts
            examples["pixel_values"] = all_images
 
            return examples

        with self.accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        # Let's first compute all the embeddings so that we can free up the text encoders
        # from memory. We will pre-compute the VAE encodings too.
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        compute_embeddings_fn = functools.partial(
            encode_prompt,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            caption_column=args.caption_column,
        )
        compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=self.vae)
        compute_rm_encodings_fn = functools.partial(
            encode_prompt_rm, 
            caption_column=args.caption_column, 
            reward_model=self.reward_model,
        )
        
        with self.accelerator.main_process_first():
            from datasets.fingerprint import Hasher

            # fingerprint used by the cache for the other processes to load the result
            # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
            new_fingerprint = Hasher.hash(args)
            new_fingerprint_for_vae = Hasher.hash(pretrained_model_name_or_path)
            new_fingerprint_for_rm = Hasher.hash(args.image_reward_version)
            train_dataset_with_embeddings = train_dataset.map(
                compute_embeddings_fn, 
                batched=True, 
                batch_size=args.mapping_batch_size,
                new_fingerprint=new_fingerprint,
            )
            train_dataset_with_vae = train_dataset.map(
                compute_vae_encodings_fn,
                batched=True,
                batch_size=args.mapping_batch_size,
                new_fingerprint=new_fingerprint_for_vae,
            )
            train_dataset_with_rm = train_dataset.map(
                compute_rm_encodings_fn,
                batched=True,
                batch_size=args.mapping_batch_size,
                new_fingerprint=new_fingerprint_for_rm,
            )
            self.precomputed_dataset = concatenate_datasets(
                [train_dataset_with_embeddings, 
                 train_dataset_with_vae.remove_columns(["image", "text", "id"]),
                 train_dataset_with_rm.remove_columns(["image", "text", "id"])], axis=1
            )
            self.precomputed_dataset = self.precomputed_dataset.with_transform(preprocess_train)

        del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
        del text_encoders, tokenizers
        gc.collect()
        torch.cuda.empty_cache()

        def collate_fn(examples):
            model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
            pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])
            
            rm_input_ids = torch.stack([torch.tensor(example["rm_input_ids"]) for example in examples])
            rm_attention_mask = torch.stack([torch.tensor(example["rm_attention_mask"]) for example in examples])
            rm_input_ids = rm_input_ids.view(-1, rm_input_ids.shape[-1])
            rm_attention_mask = rm_attention_mask.view(-1, rm_attention_mask.shape[-1])

            return {
                "model_input": model_input,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
                "rm_input_ids": rm_input_ids,
                "rm_attention_mask": rm_attention_mask,
            }

        # DataLoaders creation:
        self.train_dataloader = torch.utils.data.DataLoader(
            self.precomputed_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # Prepare everything with our `self.accelerator`.
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        if args.use_ema:
            self.ema_unet.to(self.accelerator.device)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("text2image-refl", config=vars(args))


    # time ids
    def _compute_time_ids(self, resolution, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (resolution, resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.weight_dtype)
        return add_time_ids
        
    # Function for unwrapping if torch.compile() was used in accelerate.
    def _unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def _pretrain_loss(self, args, batch) -> torch.Tensor:
        # Sample noise that we'll add to the latents
        model_input = batch["model_input"].to(self.accelerator.device)
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat(
            [self._compute_time_ids(args.resolution, s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        # Predict the noise residual
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device)
        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = model_input
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Backpropagate
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            params_to_clip = self.unet.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    
    def _reward_loss(self, args, batch) -> torch.Tensor:
        add_time_ids = torch.cat(
            [self._compute_time_ids(args.resolution, s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        # Predict the noise residual
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.accelerator.device)
        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

        latents = torch.randn((args.train_batch_size, 4, 64, 64), device=self.accelerator.device)
        
        timesteps = self.noise_scheduler.timesteps

        mid_timestep = random.randint(30, 39)

        for i, t in enumerate(timesteps[:mid_timestep]):
            with torch.no_grad():
                latent_model_input = latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        latent_model_input = latents
        latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timesteps[mid_timestep])
        noise_pred = self.unet(
            latent_model_input,
            timesteps[mid_timestep],
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
        ).sample
        pred_original_sample = self.noise_scheduler.step(noise_pred, timesteps[mid_timestep], latents).pred_original_sample.to(self.weight_dtype)
        
        pred_original_sample = 1 / self.vae.config.scaling_factor * pred_original_sample
        image = self.vae.decode(pred_original_sample.to(self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        # image encode
        def _transform():
            return transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
        rm_preprocess = _transform()
        image = rm_preprocess(image).to(self.accelerator.device)
        
        rewards = self.reward_model.score_gard(batch["rm_input_ids"], batch["rm_attention_mask"], image)
        loss = F.relu(-rewards+2)
        loss = loss.mean() * args.grad_scale

        # Backpropagate
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss
    

    def train(self, args):
        # Train!
        total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.precomputed_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f" Apply pre-trained loss? {args.apply_pre_loss}")
        logger.info(f" Apply reward loss? {args.apply_reward_loss}")
        
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * args.gradient_accumulation_steps
                first_epoch = global_step // self.num_update_steps_per_epoch
                resume_step = resume_global_step % (self.num_update_steps_per_epoch * args.gradient_accumulation_steps)


        progress_bar = tqdm(
            range(0, args.max_train_steps),
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
       
        # Record the information of best checkpoint
        if self.accelerator.is_main_process:
            best_output_dir = path
            best_train_loss = 1e9     # FIX: Load the train_loss of given `path`.


        for epoch in range(first_epoch, args.num_train_epochs):
            self.unet.train()

            # Total loss per process
            loss = 0.0

            # Pre-trained loss among all processes
            pre_avg_loss = 0.0

            # Reward loss among all processes
            reward_avg_loss = 0.0
            
            # Total loss among all processes
            train_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                logs = {}

                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                
                # Pre-train loss (NOTE: It uses same dataset as reward loss. Fix it later)
                if args.apply_pre_loss:
                    # Switch noise scheduler timesteps to original setup
                    self.noise_scheduler.reset_timesteps()
                    
                    with self.accelerator.accumulate(self.unet):
                        # Sample noise that we'll add to the latents
                        pre_loss = self._pretrain_loss(args, batch)
                        loss += pre_loss.detach().item()
                        
                        # Gather the losses across all processes for logging (if we use distributed training).
                        pre_avg_loss = self.accelerator.gather(pre_loss.repeat(args.train_batch_size)).mean()
                        train_loss += pre_avg_loss.item() / args.gradient_accumulation_steps
                                                
                    logs.update({"pre_loss": pre_loss.detach().item()})
                        
                # Reward loss
                if args.apply_reward_loss:
                    self.noise_scheduler.set_timesteps(num_inference_steps=40, 
                                                        device=self.accelerator.device)

                    with self.accelerator.accumulate(self.unet):
                        reward_loss = self._reward_loss(args, batch)
                        loss += reward_loss.detach().item()

                        # Gather the losses across all processes for logging (if we use distributed training).
                        reward_avg_loss = self.accelerator.gather(reward_loss.repeat(args.train_batch_size)).mean()
                        train_loss += reward_avg_loss.item() / args.gradient_accumulation_steps

                    logs.update({"reward_loss": reward_loss.detach().item()})

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if args.use_ema:
                        self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss, 
                                          "pre_loss": pre_avg_loss, 
                                          "reward_loss": reward_avg_loss}, step=global_step)

                    # Store the model weights
                    if self.accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                            if args.save_only_one_ckpt:
                                if train_loss < best_train_loss:
                                    
                                    best_train_loss = train_loss
                                    logger.info(f"Best loss changed to {best_train_loss}")
                                    
                                    # Delete previous best checkpoint 
                                    if best_output_dir is not None:
                                        shutil.rmtree(best_output_dir)
                                        logger.info(f"Removed previous state at {best_output_dir}")
                                    
                                    best_output_dir = save_path

                                    self.accelerator.save_state(save_path)                            
                                    logger.info(f"Saved state to {save_path}")
                            else:
                                self.accelerator.save_state(save_path)                            
                                logger.info(f"Saved state to {save_path}")
                    
                    train_loss = 0.0

                logs.update({"step_loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]})
                progress_bar.set_postfix(**logs)

                loss = 0.0

                if global_step >= args.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self._unwrap_model(self.unet)
            if args.use_ema:
                self.ema_unet.copy_to(self.unet.parameters())

            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=self.unet,
                vae=self.vae,
                revision=args.revision,
                torch_dtype=self.weight_dtype,
                cache_dir=args.cache_dir
            )
            if args.prediction_type is not None:
                scheduler_args = {"prediction_type": args.prediction_type}
                pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
            pipeline.save_pretrained(args.output_dir)

        self.accelerator.end_training()