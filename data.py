import argparse
import logging
import os
import shutil
from pathlib import Path
# import accelerate
# import datasets
# import transformers
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.state import AcceleratorState
# from accelerate.utils import ProjectConfiguration, set_seed
# from huggingface_hub import create_repo
# from packaging import version
# from safetensors import safe_open
# from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
# from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
# from transformers.utils import ContextManagers
# from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# from unet import UNet2DConditionModel
import diffusers
# from diffusers import AutoencoderKL, StableDiffusionPipeline, LCMScheduler, EMAModel
# from diffusers.optimization import get_scheduler
# from diffusers.utils import check_min_version, deprecate
# from diffusers.utils.import_utils import is_xformers_available
# from diffusers.utils.torch_utils import is_compiled_module

# from torch import nn
# import math
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_annotations(annots_file):
    annots_dict = {}
    if not os.path.exists(annots_file):
        return annots_dict
    folder_name = os.path.basename(os.path.dirname(annots_file))
    root_dir = os.path.dirname(os.path.dirname(annots_file))
    with open(annots_file, 'r') as file:
        annots = json.load(file)
        for k, v in annots.items():
            file_path = os.path.join(root_dir, folder_name, "frames", k)
            idxs = [i for i, x in enumerate(v) if "11" in x and isinstance(x["11"], str)]
            if len(idxs) == 0:
                continue
            if len(idxs) > 1:
                idx = random.choice(idxs)
            else:
                idx = idxs[0]
            annot = v[idx]["11"]
            if not isinstance(annot, str):
                continue
            annots_dict[file_path] = annot
    return annots_dict


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, frames_dirname="frames", annots_file_name="txts_flash_yoso.json", transform=None,
                    resolution=512):
        # self.data = []
        self.resolution = resolution
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'runwayml/stable-diffusion-v1-5', subfolder="tokenizer", )
        if isinstance(root_dir, (str, Path)):
            self.root_dirs = [root_dir]
        elif isinstance(root_dir, (list, tuple)):
            self.root_dirs = root_dir
        self.file_paths = []
        self.annots_dict = {}
        self.no_annots_file_paths = []
        self.multiple_annots_file_paths = []
        futures = []
        with ThreadPoolExecutor(128) as executor:
            for root_dir in self.root_dirs:
                subdirs = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
                for subdir in subdirs:
                    annots_file = os.path.join(root_dir, subdir, annots_file_name)
                    futures.append(executor.submit(read_annotations, annots_file))
        for future in as_completed(futures):
            annots_dict = future.result()
            self.annots_dict.update(annots_dict)
            self.file_paths.extend(list(annots_dict.keys()))
        print(f"Loaded {len(self.file_paths)} annotations")
        # print(f"Loaded {len(self.no_annots_file_paths)} files with no annotations")
        # print(f"Loaded {len(self.multiple_annots_file_paths)} files with multiple annotations")
        # print(f"Total files: {len(self.file_paths) + len(self.no_annots_file_paths) + len(self.multiple_annots_file_paths)}")

    # def __init__(self, root_dir, frames_dirname="frames", annots_file_name="txts_flash_yoso.json", transform=None,
    #              resolution=512):
    #     # self.data = []
    #     if transform is None:
    #         transform = transforms.Compose([
    #             transforms.Resize((resolution, resolution)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ])
    #     self.transform = transform
    #     self.resolution = resolution
    #     self.tokenizer = CLIPTokenizer.from_pretrained(
    #         'runwayml/stable-diffusion-v1-5', subfolder="tokenizer", )
    #     self.root_dir = root_dir
    #     self.annots_files = glob.glob(os.path.join(root_dir, "*", annots_file_name))
    #     self.file_paths = []
    #     self.annots_dict = {}
    #     for annots_file in self.annots_files:
    #         folder_name = os.path.basename(os.path.dirname(annots_file))
    #         with open(annots_file, 'r') as file:
    #             annots = json.load(file)
    #             for k, v in annots.items():
    #                 file_path = os.path.join(root_dir, folder_name, frames_dirname, k)
    #                 annot = v[0]["11"]
    #                 if not isinstance(annot, str):
    #                     continue
    #                 self.annots_dict[file_path] = annot
    #                 self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        text = self.annots_dict[file_path]
        prompt = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length",
                                truncation=True, return_tensors="pt").input_ids
        image = Image.open(file_path).convert('RGB')
        image = ImageOps.pad(image, (self.resolution, self.resolution))
        if self.transform:
            image = self.transform(image)

        return text, prompt, image


def main():
    data_path = "/mnt/data"
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    root_dir = [f"{data_path}/fashion/fashion-shops",
                   f"{data_path}/fashion/fashion-shops-jul/fashion/fashion-shops-mini",
                   f"{data_path}/fashion/fashion-shops-jul/fashion/fashion-shops-random",
                   f"{data_path}/fashion/pexels/download",
                   f"{data_path}/fashion/jul/shops/fashion-shops-plus-size-new",
                   f"{data_path}/fashion/feb/shops/fashion-shops-plus-size-unfiltered" ]
    annots_file_name = "background_txts_flash5.json"

    dataset = CustomImageDataset(root_dir=root_dir, transform=transform, resolution=resolution,
                                 annots_file_name=annots_file_name)

    print(f"len(dataset): {len(dataset)}")

if __name__ == "__main__":
    main()
