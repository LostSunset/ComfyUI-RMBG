# ComfyUI-RMBG v2.0.0
#
# This node facilitates background removal using various models, including RMBG-2.0, INSPYRENET, BEN, BEN2, and BIREFNET-HR.
# It utilizes advanced deep learning techniques to process images and generate accurate masks for background removal.
#
# AILab Image and Mask Tools
# This module is specifically designed for ComfyUI-RMBG, enhancing workflows within ComfyUI.
# It offers a collection of utility nodes for efficient handling of images and masks:
#
# 1. Preview Nodes:
#    - AiLab_Preview: A universal preview tool for both images and masks.
#    - AiLab_ImagePreview: A specialized preview tool for images.
#    - AiLab_MaskPreview: A specialized preview tool for masks.
#    - AiLab_LoadImage: A node for loading images with some Frequently used options.
#
# These nodes are crafted to streamline common image and mask operations within ComfyUI workflows.
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-RMBG

import os
import random
import folder_paths
import numpy as np
import hashlib
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageSequence, ImageChops
import torchvision.transforms.functional as T
from scipy import ndimage

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2mask(image):
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

def blend_overlay(img_1, img_2):
    arr1 = np.array(img_1).astype(float) / 255.0
    arr2 = np.array(img_2).astype(float) / 255.0
    mask = arr2 < 0.5
    result = np.zeros_like(arr1)
    result[mask] = 2 * arr1[mask] * arr2[mask]
    result[~mask] = 1 - 2 * (1 - arr1[~mask]) * (1 - arr2[~mask])
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))

# Base class for preview
class AiLab_PreviewBase:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""

    def get_unique_filename(self, filename_prefix):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename_prefix + self.prefix_append
        counter = 1
        while True:
            file = f"{filename}_{counter:04d}.png"
            full_path = os.path.join(self.output_dir, file)
            if not os.path.exists(full_path):
                return full_path, file
            counter += 1

    def save_image(self, image, filename_prefix, prompt=None, extra_pnginfo=None):
        results = []
      
        try:
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 4:  # Batch of images
                    for i in range(image.shape[0]):
                        full_output_path, file = self.get_unique_filename(filename_prefix)
                        img = Image.fromarray(np.clip(image[i].cpu().numpy() * 255, 0, 255).astype(np.uint8))
                        img.save(full_output_path)
                        results.append({"filename": full_output_path, "subfolder": "", "type": self.type})
                else:
                    full_output_path, file = self.get_unique_filename(filename_prefix)
                    img = Image.fromarray(np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))
                    img.save(full_output_path)
                    results.append({"filename": full_output_path, "subfolder": "", "type": self.type})
            else:
                full_output_path, file = self.get_unique_filename(filename_prefix)
                image.save(full_output_path)
                results.append({"filename": full_output_path, "subfolder": "", "type": self.type})
            
            return {
                "ui": {"images": results},
            }
        except Exception as e:
            print(f"Error saving image: {e}")
            return {"ui": {}}

# Preview node
class AiLab_Preview(AiLab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🛠️UTIL/🖼️IMAGE"

    def preview(self, image=None, mask=None, prompt=None, extra_pnginfo=None):
        results = []
        
        if image is not None:
            image_result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
            if "ui" in image_result and "images" in image_result["ui"]:
                results.extend(image_result["ui"]["images"])
        
        if mask is not None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            mask_result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
            if "ui" in mask_result and "images" in mask_result["ui"]:
                results.extend(mask_result["ui"]["images"])
        
        return {
            "ui": {"images": results},
            "result": (image if image is not None else None, mask if mask is not None else None)
        }

# Mask preview node
class AiLab_MaskPreview(AiLab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_mask_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",),},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "preview_mask"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🛠️UTIL/🖼️IMAGE"

    def preview_mask(self, mask, prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (mask,)
        }

# Image preview node
class AiLab_ImagePreview(AiLab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_image_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",),},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "preview_image"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🛠️UTIL/🖼️IMAGE"

    def preview_image(self, image, prompt=None, extra_pnginfo=None):
        result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (image,)
        }

# Image loader node
class AiLab_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'))]
        return {
            "required": {
                "image": (sorted(files) or [""], {"image_upload": True}),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha", "tooltip": "Select channel to extract mask from"}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "Scale image by this factor (ignored if longest_side > 0)"}),
                "longest_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8, "tooltip": "Resize image so longest side equals this value (0 = disabled)"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "🧪AILab/🛠️UTIL/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    def load_image(self, image, mask_channel="alpha", scale_by=1.0, longest_side=0, extra_pnginfo=None):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            img = Image.open(image_path)
            
            orig_width, orig_height = img.size
            if longest_side > 0:
                if orig_width >= orig_height:
                    new_width = longest_side
                    new_height = int(orig_height * (longest_side / orig_width))
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                else:
                    new_height = longest_side
                    new_width = int(orig_width * (longest_side / orig_height))
                img = img.resize((new_width, new_height), Image.LANCZOS)
            elif scale_by != 1.0:
                new_width = int(orig_width * scale_by)
                new_height = int(orig_height * scale_by)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            width, height = img.size
            
            output_images = []
            output_masks = []
            for i in ImageSequence.Iterator(img):
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                if mask_channel == "alpha" and 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                elif mask_channel == "red" and 'R' in i.getbands():
                    mask = np.array(i.getchannel('R')).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask)
                elif mask_channel == "green" and 'G' in i.getbands():
                    mask = np.array(i.getchannel('G')).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask)
                elif mask_channel == "blue" and 'B' in i.getbands():
                    mask = np.array(i.getchannel('B')).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask)
                else:
                    mask = torch.ones((height, width), dtype=torch.float32, device="cpu")
                
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))
            
            if len(output_images) > 1:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]
            
            mask_image = output_mask.reshape((-1, 1, output_mask.shape[-2], output_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            
            return (output_image, output_mask, mask_image, width, height)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading image: {e}")
            empty_image = torch.zeros(1, 3, 64, 64)
            empty_mask = torch.zeros(1, 64, 64)
            empty_mask_image = empty_mask.reshape((-1, 1, 64, 64)).movedim(1, -1).expand(-1, -1, -1, 3)
            return (empty_image, empty_mask, empty_mask_image, 64, 64)
    
    @classmethod
    def IS_CHANGED(cls, image, mask_channel="alpha", scale_by=1.0, longest_side=0, extra_pnginfo=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, mask_channel="alpha", scale_by=1.0, longest_side=0, extra_pnginfo=None):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        
        return True



# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AiLab_LoadImage": AiLab_LoadImage,
    "AiLab_Preview": AiLab_Preview,
    "AiLab_ImagePreview": AiLab_ImagePreview,
    "AiLab_MaskPreview": AiLab_MaskPreview,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AiLab_LoadImage": "Load Image (RMBG) 🖼️",
    "AiLab_Preview": "Preview (RMBG) 🖼️🎭",
    "AiLab_ImagePreview": "Image Preview (RMBG) 🖼️",
    "AiLab_MaskPreview": "Mask Preview (RMBG) 🎭",
} 