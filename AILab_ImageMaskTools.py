# ComfyUI-RMBG v2.3.1
#
# This node facilitates background removal using various models, including RMBG-2.0, INSPYRENET, BEN, BEN2, and BIREFNET-HR.
# It utilizes advanced deep learning techniques to process images and generate accurate masks for background removal.
#
# AILab Image and Mask Tools
# This module is specifically designed for ComfyUI-RMBG, enhancing workflows within ComfyUI.
# It offers a collection of utility nodes for efficient handling of images and masks:
#
# 1. Preview Nodes:
#    - Preview: A universal preview tool for both images and masks.
#    - ImagePreview: A specialized preview tool for images.
#    - MaskPreview: A specialized preview tool for masks.
#    - LoadImage: A node for loading images with some Frequently used options.
#
# 2. Conversion Node:
#    - ImageMaskConvert: Converts between image and mask formats and extracts masks from image channels.
#
# 3. Mask Processing Nodes:
#    - MaskEnhancer: Refines masks through techniques such as blur, smoothing, expansion/contraction, and hole filling.
#    - MaskCombiner: Combines multiple masks using union, intersection, or difference operations.
#
# 4. Image Processing Nodes:
#    - ImageCombiner: Combines foreground and background images with various blending modes and positioning options.
#    - ImageStitch: Stitches multiple images together in various directions.
#    - ImageCrop: Crops an image to a specified size and position.
#    - ICLoRAConcat: Concatenates images with a mask using IC LoRA.

# These nodes are crafted to streamline common image and mask operations within ComfyUI workflows.

import os
import random
import folder_paths
import numpy as np
import hashlib
import torch
import cv2
from nodes import MAX_RESOLUTION
from PIL import Image, ImageFilter, ImageOps, ImageSequence, ImageChops
import torchvision.transforms.functional as T
from comfy.utils import common_upscale
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

def fill_mask(width, height, mask, box=(0, 0), color=0):
    bg = Image.new("L", (width, height), color)
    bg.paste(mask, box, mask)
    return bg

def empty_image(width, height, batch_size=1):
    return torch.zeros([batch_size, height, width, 3])

def upscale_mask(mask, width, height):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = common_upscale(mask, width, height, 'bicubic', 'disabled')
    mask = mask.squeeze(1)
    return mask

def extract_alpha_mask(image):
    alpha = image[..., 3]
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    if len(alpha.shape) == 4:
        alpha = alpha[:, :, :, 0]
    return alpha.unsqueeze(1) if alpha.ndim == 3 else alpha

def ensure_mask_shape(mask):
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask.squeeze(1)
    return mask

# Base class for preview
class AILab_PreviewBase:
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
                        results.append({"filename": file, "subfolder": "", "type": self.type})
                else:
                    full_output_path, file = self.get_unique_filename(filename_prefix)
                    img = Image.fromarray(np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))
                    img.save(full_output_path)
                    results.append({"filename": file, "subfolder": "", "type": self.type})
            else:
                full_output_path, file = self.get_unique_filename(filename_prefix)
                image.save(full_output_path)
                results.append({"filename": file, "subfolder": "", "type": self.type})

            return {
                "ui": {"images": results},
            }
        except Exception as e:
            print(f"Error saving image: {e}")
            return {"ui": {}}

# Preview node
class AILab_Preview(AILab_PreviewBase):
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
    CATEGORY = "🧪AILab/🖼️IMAGE"

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
class AILab_MaskPreview(AILab_PreviewBase):
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
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview_mask(self, mask, prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (mask,)
        }

# Image preview node
class AILab_ImagePreview(AILab_PreviewBase):
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
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview_image(self, image, prompt=None, extra_pnginfo=None):
        result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (image,)
        }

# Image mask conversion node
class AILab_ImageMaskConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "convert"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def convert(self, image=None, mask=None, mask_channel="alpha"):
        # Case 1: No inputs
        if image is None and mask is None:
            empty_image = torch.zeros(1, 3, 64, 64)
            empty_mask = torch.zeros(1, 64, 64)
            return (empty_image, empty_mask)
            
        # Case 2: Only mask input
        if image is None and mask is not None:
            if mask.ndim == 4:
                tensor = mask.permute(0, 2, 3, 1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask)
            elif mask.ndim == 3:
                tensor = mask.unsqueeze(-1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask)
            elif mask.ndim == 2:
                tensor = mask.unsqueeze(0).unsqueeze(-1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask.unsqueeze(0))
            else:
                print(f"Invalid mask shape: {mask.shape}")
                empty_image = torch.zeros(1, 3, 64, 64)
                return (empty_image, mask)
            
        # Case 3: Only image input
        if image is not None and mask is None:
            mask_list = []
            for img in image:
                pil_img = tensor2pil(img)
                pil_img = pil_img.convert("RGBA")
                r, g, b, a = pil_img.split()
                if mask_channel == "red":
                    channel_img = r
                elif mask_channel == "green":
                    channel_img = g
                elif mask_channel == "blue":
                    channel_img = b
                elif mask_channel == "alpha":
                    channel_img = a
                mask = np.array(channel_img.convert("L")).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask)
                mask_list.append(mask_tensor)
            result_mask = torch.stack(mask_list)
            return (image, result_mask)

        if image is not None and mask is not None:
            if mask.ndim == 4:  # [B,C,H,W]
                mask = mask.squeeze(1)  # Convert to [B,H,W]
            return (image, mask)

# Mask enhancer node
class AILab_MaskEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "mask": "Input mask to be processed.",
            "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "smooth": "Smooth the mask edges (0 for no smoothing, higher values create smoother edges).",
            "fill_holes": "Enable to fill holes in the mask.",
            "invert_output": "Enable to invert the mask output (useful for certain effects)."
        }
        
        return {
            "required": {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5, "tooltip": tooltips["smooth"]}),
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": tooltips["fill_holes"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def fill_mask_region(self, mask_pil):
        """Fill holes in the mask"""
        mask_np = np.array(mask_pil)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask_np)
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], 0, 255, -1)  # -1 means fill
        return Image.fromarray(filled_mask)

    def process_mask(self, mask, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=0.0, 
                    fill_holes=False, invert_output=False):
        processed_masks = []
        
        for mask_item in mask:
            m = mask_item * (1 + (1 - sensitivity))
            m = torch.clamp(m, 0, 1)
            
            if smooth > 0:
                mask_np = m.cpu().numpy()
                binary_mask = (mask_np > 0.5).astype(np.float32)
                blurred_mask = ndimage.gaussian_filter(binary_mask, sigma=smooth)
                final_mask = (blurred_mask > 0.5).astype(np.float32)
                m = torch.from_numpy(final_mask)
            
            if fill_holes:
                mask_pil = tensor2pil(m)
                mask_pil = self.fill_mask_region(mask_pil)
                m = pil2tensor(mask_pil).squeeze(0)
            
            if mask_blur > 0:
                mask_pil = tensor2pil(m)
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                m = pil2tensor(mask_pil).squeeze(0)
            
            if mask_offset != 0:
                mask_pil = tensor2pil(m)
                if mask_offset > 0:
                    for _ in range(mask_offset):
                        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
                else:
                    for _ in range(-mask_offset):
                        mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
                m = pil2tensor(mask_pil).squeeze(0)
            
            if invert_output:
                m = 1.0 - m
            
            processed_masks.append(m.unsqueeze(0))
        
        return (torch.cat(processed_masks, dim=0),)

# Mask combiner node
class AILab_MaskCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mode": (["combine", "intersection", "difference"], {"default": "combine"})
            },
            "optional": {
                "mask_2": ("MASK", {"default": None}),
                "mask_3": ("MASK", {"default": None}),
                "mask_4": ("MASK", {"default": None})
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_masks"

    def combine_masks(self, mask_1, mode="combine", mask_2=None, mask_3=None, mask_4=None):
        try:
            masks = [m for m in [mask_1, mask_2, mask_3, mask_4] if m is not None]
            
            if len(masks) <= 1:
                return (masks[0] if masks else torch.zeros((1, 64, 64), dtype=torch.float32),)
                
            ref_shape = masks[0].shape
            masks = [self._resize_if_needed(m, ref_shape) for m in masks]
            
            if mode == "combine":
                result = torch.maximum(masks[0], masks[1])
                for mask in masks[2:]:
                    result = torch.maximum(result, mask)
            elif mode == "intersection":
                result = torch.minimum(masks[0], masks[1])
            else:
                result = torch.abs(masks[0] - masks[1])
                
            return (torch.clamp(result, 0, 1),)
        except Exception as e:
            print(f"Error in combine_masks: {str(e)}")
            print(f"Mask shapes: {[m.shape for m in masks]}")
            raise e
    
    def _resize_if_needed(self, mask, target_shape):
        try:
            if mask.shape == target_shape:
                return mask
                
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            elif len(mask.shape) == 4:
                mask = mask.squeeze(1)
            
            target_height = target_shape[-2] if len(target_shape) >= 2 else target_shape[0]
            target_width = target_shape[-1] if len(target_shape) >= 2 else target_shape[1]
            
            resized_masks = []
            for i in range(mask.shape[0]):
                mask_np = mask[i].cpu().numpy()
                img = Image.fromarray((mask_np * 255).astype(np.uint8))
                img_resized = img.resize((target_width, target_height), Image.LANCZOS)
                mask_resized = np.array(img_resized).astype(np.float32) / 255.0
                resized_masks.append(torch.from_numpy(mask_resized))
            
            return torch.stack(resized_masks)
            
        except Exception as e:
            print(f"Error in _resize_if_needed: {str(e)}")
            print(f"Input mask shape: {mask.shape}, Target shape: {target_shape}")
            raise e

# Image loader node
class AILab_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'))]
        return {
            "required": {
                "image": (sorted(files) or [""], {"image_upload": True}),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha", "tooltip": "Select channel to extract mask from"}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "Scale image by this factor (ignored if size > 0)"}),
                "resize_mode": (["longest_side", "shortest_side", "width", "height"], {"default": "longest_side", "tooltip": "Choose how to resize the image"}),
                "size": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Target size for the selected resize mode (0 = keep original size)"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    def load_image(self, image, mask_channel="alpha", scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            img = Image.open(image_path)
            
            orig_width, orig_height = img.size
            
            # Image resizing logic
            if size > 0:
                if resize_mode == "longest_side":
                    if orig_width >= orig_height:
                        new_width = size
                        new_height = int(orig_height * (size / orig_width))
                    else:
                        new_height = size
                        new_width = int(orig_width * (size / orig_height))
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                elif resize_mode == "shortest_side":
                    if orig_width <= orig_height:
                        new_width = size
                        new_height = int(orig_height * (size / orig_width))
                    else:
                        new_height = size
                        new_width = int(orig_width * (size / orig_height))
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                elif resize_mode == "width":
                    new_width = size
                    new_height = int(orig_height * (size / orig_width))
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                elif resize_mode == "height":
                    new_height = size
                    new_width = int(orig_width * (size / orig_height))
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
    def IS_CHANGED(cls, image, mask_channel="alpha", scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(cls, image, mask_channel="alpha", scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        
        return True

# Image combiner node
class AILab_ImageCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "mode": (["normal", "multiply", "screen", "overlay", "add", "subtract"], 
                              {"default": "normal"}),
                "foreground_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "position_x": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "position_y": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "foreground_mask": ("MASK", {"default": None}),
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine_images"
    
    def combine_images(self, foreground, background, mode="normal", foreground_opacity=1.0, 
                      foreground_scale=1.0, position_x=50, position_y=50, foreground_mask=None):
        if len(foreground.shape) == 3:
            foreground = foreground.unsqueeze(0)
        if len(background.shape) == 3:
            background = background.unsqueeze(0)
        
        batch_size = foreground.shape[0]
        output_images = []
        
        for b in range(batch_size):
            fg_pil = tensor2pil(foreground[b])
            bg_pil = tensor2pil(background[b])
            
            if fg_pil.mode != 'RGBA':
                fg_pil = fg_pil.convert('RGBA')
            
            if foreground_scale != 1.0:
                new_width = int(fg_pil.width * foreground_scale)
                new_height = int(fg_pil.height * foreground_scale)
                fg_pil = fg_pil.resize((new_width, new_height), Image.LANCZOS)
            
            if foreground_mask is not None:
                mask_tensor = foreground_mask[b] if len(foreground_mask.shape) > 2 else foreground_mask
                mask_pil = Image.fromarray(np.uint8(mask_tensor.cpu().numpy() * 255))
                if mask_pil.size != fg_pil.size:
                    mask_pil = mask_pil.resize(fg_pil.size, Image.LANCZOS)
                r, g, b, a = fg_pil.split()
                a = ImageChops.multiply(a, mask_pil)
                fg_pil = Image.merge('RGBA', (r, g, b, a))
            
            fg_w, fg_h = fg_pil.size
            bg_w, bg_h = bg_pil.size
            
            x = int(bg_w * position_x / 100 - fg_w / 2)
            y = int(bg_h * position_y / 100 - fg_h / 2)
            
            new_fg = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
            new_fg.paste(fg_pil, (x, y), fg_pil)
            fg_pil = new_fg
            
            if bg_pil.mode != 'RGBA':
                bg_pil = bg_pil.convert('RGBA')
            
            if foreground_opacity < 1.0:
                r, g, b, a = fg_pil.split()
                a = Image.eval(a, lambda x: int(x * foreground_opacity))
                fg_pil = Image.merge('RGBA', (r, g, b, a))
            
            if mode == "normal":
                result = bg_pil.copy()
                result = Image.alpha_composite(result, fg_pil)
            else:
                alpha = fg_pil.split()[3]
                fg_rgb = fg_pil.convert('RGB')
                bg_rgb = bg_pil.convert('RGB')
                
                if mode == "multiply":
                    blended = ImageChops.multiply(fg_rgb, bg_rgb)
                elif mode == "screen":
                    blended = ImageChops.screen(fg_rgb, bg_rgb)
                elif mode == "add":
                    blended = ImageChops.add(fg_rgb, bg_rgb, 1.0)
                elif mode == "subtract":
                    blended = ImageChops.subtract(fg_rgb, bg_rgb, 1.0)
                elif mode == "overlay":
                    blended = blend_overlay(fg_rgb, bg_rgb)
                else:
                    blended = fg_rgb
                
                blended = blended.convert('RGBA')
                r, g, b, _ = blended.split()
                blended = Image.merge('RGBA', (r, g, b, alpha))
                result = bg_pil.copy()
                result = Image.alpha_composite(result, blended)
            
            if result.mode != 'RGB':
                white_bg = Image.new('RGB', result.size, 'white')
                result = Image.alpha_composite(white_bg.convert('RGBA'), result)
                result = result.convert('RGB')
            
            output_images.append(pil2tensor(result))
        
        return (torch.cat(output_images, dim=0),)

# Mask extractor node
class AILab_MaskExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["extract_masked_area", "apply_mask", "invert_mask"], {"default": "invert_mask"}),
                "background": (["transparent", "black", "white", "original"], {"default": "transparent"})
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_masked_area"

    def _prepare_mask(self, mask_np, image_shape):
        try:
            if isinstance(mask_np, torch.Tensor):
                mask_np = mask_np.cpu().numpy()
            mask_np = np.array(mask_np)
            while len(mask_np.shape) > 2 and mask_np.shape[-1] == 1:
                mask_np = mask_np.squeeze(-1)
            while len(mask_np.shape) > 2 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()
            if mask_np.shape != image_shape[:2]:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask_np = mask_np[..., np.newaxis]
            mask_np = np.repeat(mask_np, image_shape[2], axis=2)
            return mask_np
        except Exception as e:
            print(f"Error in _prepare_mask: {str(e)}")
            raise e

    def extract_masked_area(self, image, mask, mode="extract_masked_area", background="transparent"):
        try:
            pil_image = tensor2pil(image)
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            mask_np = self._prepare_mask(mask, image_np.shape)
            result_np = np.zeros_like(image_np)
            
            if mode == "extract_masked_area":
                result_np = image_np * mask_np
                if background == "transparent":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * mask_np
                    result_rgba[:, :, 3] = mask_np[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "black":
                    pass  # Already done with image_np * mask_np
                elif background == "white":
                    result_np = result_np + (1 - mask_np)
                elif background == "original":
                    result_np = image_np * mask_np
            
            elif mode == "apply_mask":
                result_np = image_np * mask_np
                if background == "transparent":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * mask_np
                    result_rgba[:, :, 3] = mask_np[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "white":
                    result_np = result_np + (1 - mask_np)
                elif background == "original":
                    result_np = image_np * mask_np + image_np * (1 - mask_np)
            
            elif mode == "invert_mask":
                result_np = image_np * (1 - mask_np)
                if background == "transparent":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * (1 - mask_np)
                    result_rgba[:, :, 3] = (1 - mask_np)[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "white":
                    result_np = result_np + mask_np
                elif background == "original":
                    result_np = image_np * (1 - mask_np) + image_np * mask_np
            
            result_pil = Image.fromarray(np.clip(result_np * 255, 0, 255).astype(np.uint8))
            return (pil2tensor(result_pil),)
        except Exception as e:
            print(f"Error in extract_masked_area: {str(e)}")
            raise e

# Image Stitch node
class AILab_ImageStitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "concat_direction": (['right', 'top', 'left', 'bottom'], {"default": 'right'}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch_images"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def stitch_images(self, image1, image2, concat_direction):
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            image1 = image1.repeat(max_batch // image1.shape[0], 1, 1, 1)
            image2 = image2.repeat(max_batch // image2.shape[0], 1, 1, 1)

        if concat_direction in ['right', 'left']:
            # Match heights for horizontal stitching
            h1 = image1.shape[1]
            h2, w2 = image2.shape[1:3]
            aspect = w2 / h2
            
            new_h = h1
            new_w = int(h1 * aspect)
            
            image2 = self._resize(image2, new_w, new_h)
        else:
            # Match widths for vertical stitching
            w1 = image1.shape[2]
            h2, w2 = image2.shape[1:3]
            aspect = h2 / w2
            
            new_w = w1
            new_h = int(w1 * aspect)
            
            image2 = self._resize(image2, new_w, new_h)

        ch1, ch2 = image1.shape[-1], image2.shape[-1]
        if ch1 != ch2:
            if ch1 < ch2:
                image1 = torch.cat((image1, torch.ones((*image1.shape[:-1], ch2-ch1), device=image1.device)), dim=-1)
            else:
                image2 = torch.cat((image2, torch.ones((*image2.shape[:-1], ch1-ch2), device=image2.device)), dim=-1)

        if concat_direction == 'right':
            result = torch.cat((image1, image2), dim=2)
        elif concat_direction == 'bottom':
            result = torch.cat((image1, image2), dim=1)
        elif concat_direction == 'left':
            result = torch.cat((image2, image1), dim=2)
        elif concat_direction == 'top':
            result = torch.cat((image2, image1), dim=1)
            
        return (result,)

    def _resize(self, image, width, height):
        img = image.movedim(-1, 1)
        resized = common_upscale(img, width, height, "lanczos", "disabled")
        return resized.movedim(1, -1)

# Image Crop node
class AILab_ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Width of the crop region in pixels. Will be clamped to image width."}),
                "height": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Height of the crop region in pixels. Will be clamped to image height."}),
                "x_offset": ("INT", {"default": 0, "min": -99999, "step": 1, "tooltip": "Horizontal offset (in pixels) added to the crop position. Positive values move right, negative left."}),
                "y_offset": ("INT", {"default": 0, "min": -99999, "step": 1, "tooltip": "Vertical offset (in pixels) added to the crop position. Positive values move down, negative up."}),
                "split": ("BOOLEAN", {"default": False, "tooltip": "If True, output the cropped region and the rest of the image with the crop area set to zero. If False, the rest is a zero image."}),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"], {"tooltip": "Anchor position for the crop region. Determines where the crop is placed relative to the image."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("CROP", "REST")
    FUNCTION = "execute"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def execute(self, image, width, height, position, x_offset, y_offset, split=False):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        crop = image[:, y:y2, x:x2, :]
        rest = None
        if split:
            top = image[:, 0:y, :, :] if y > 0 else None
            bottom = image[:, y2:oh, :, :] if y2 < oh else None
            left = image[:, y:y2, 0:x, :] if x > 0 else None
            right = image[:, y:y2, x2:ow, :] if x2 < ow else None

            parts = []
            if top is not None:
                parts.append(top)
            if left is not None or right is not None:
                row_parts = []
                if left is not None:
                    row_parts.append(left)
                if right is not None:
                    row_parts.append(right)
                if row_parts:
                    row = torch.cat(row_parts, dim=2)
                    parts.append(row)
            if bottom is not None:
                parts.append(bottom)
            if parts:
                rest = torch.cat(parts, dim=1)
            else:
                rest = torch.zeros_like(image[:, :0, :0, :])
        else:
            rest = image.clone()
            rest[:] = 0
        return (crop, rest)

# ICLoRA Concat node
class AILab_ICLoRAConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_image": ("IMAGE",{"tooltip": ("The main image to be used as the foreground (object) in the concatenation.\nIf the image has 4 channels (RGBA), the alpha channel will be automatically extracted and used as the object mask if no mask is provided.")}),
                "layout": (["top-bottom", "left-right"], {"default": "left-right", "tooltip": "The direction in which to concatenate the images: top-bottom or left-right."}),
                "custom_size": ("INT", {"default": 0, "max": MAX_RESOLUTION, "min": 0, "step": 8, "tooltip": "If 0, the output image size is unchanged. Otherwise, sets the base image height (for left-right) or base image width (for top-bottom) in pixels for the concatenation. The object image will be scaled proportionally to match the base image in the concatenation direction."}),
            },
            "optional": {
                "object_mask": ("MASK", {"tooltip": "Mask for the object_image. Defines the region of the object_image to be blended into the base_image."}),
                "base_image": ("IMAGE", {"tooltip": "The background image to be concatenated with the object_image.\nIf the image has 4 channels (RGBA), the alpha channel will be automatically extracted and used as the base mask if no mask is provided."}),
                "base_mask": ("MASK", {"tooltip": "Mask for the base_image. Defines the region of the base_image to be blended with the object_image."}),
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    FUNCTION = "create"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "OBJECT_MASK", "BASE_MASK", "WIDTH", "HEIGHT", "X", "Y")

    def create(self, object_image, layout, custom_size=0, base_image=None, object_mask=None, base_mask=None):
        if object_image.shape[-1] == 4 and object_mask is None:
            object_mask = extract_alpha_mask(object_image)
            object_image = object_image[..., :3]
        if base_image is not None and base_image.shape[-1] == 4 and base_mask is None:
            base_mask = extract_alpha_mask(base_image)
            base_image = base_image[..., :3]

        if base_image is None:
            base_image = empty_image(object_image.shape[2], object_image.shape[1])
            base_mask = torch.full((1, object_image.shape[1], object_image.shape[2]), 1, dtype=torch.float32, device="cpu")
        elif base_image is not None and base_mask is None:
            raise ValueError("base_mask is required when base_image is provided")

        object_mask = ensure_mask_shape(object_mask)
        base_mask = ensure_mask_shape(base_mask)

        _, base_h, base_w, base_c = base_image.shape
        _, obj_h, obj_w, obj_c = object_image.shape

        if layout == 'left-right':
            if custom_size > 0:
                new_base_h = custom_size
                new_base_w = int(base_w * (custom_size / base_h))
                base_image = base_image.movedim(-1, 1)
                base_image = common_upscale(base_image, new_base_w, new_base_h, 'bicubic', 'disabled')
                base_image = base_image.movedim(1, -1)
                if base_mask is not None:
                    base_mask = upscale_mask(base_mask, new_base_w, new_base_h)
                base_h, base_w = new_base_h, new_base_w

            scale = base_h / obj_h
            new_obj_w = int(obj_w * scale)
            object_image = object_image.movedim(-1, 1)
            object_image = common_upscale(object_image, new_obj_w, base_h, 'bicubic', 'disabled')
            object_image = object_image.movedim(1, -1)
            if object_mask is not None:
                object_mask = upscale_mask(object_mask, new_obj_w, base_h)
            else:
                object_mask = torch.full((1, base_h, new_obj_w), 1, dtype=torch.float32, device="cpu")

            if object_image.shape[-1] != base_image.shape[-1]:
                min_c = min(object_image.shape[-1], base_image.shape[-1])
                object_image = object_image[..., :min_c]
                base_image = base_image[..., :min_c]
            
            image = torch.cat((object_image, base_image), dim=2)
            batch = object_mask.shape[0]
            out_h = base_h
            out_w = new_obj_w + base_w
            object_mask_resized = object_mask
            base_mask_resized = base_mask
            
            if object_mask_resized.shape[-2:] != (base_h, new_obj_w):
                object_mask_resized = upscale_mask(object_mask_resized, new_obj_w, base_h)
            if base_mask_resized.shape[-2:] != (base_h, base_w):
                base_mask_resized = upscale_mask(base_mask_resized, base_w, base_h)
            
            OBJECT_MASK = torch.zeros((batch, out_h, out_w), dtype=object_mask_resized.dtype, device=object_mask_resized.device)
            BASE_MASK = torch.zeros((batch, out_h, out_w), dtype=base_mask_resized.dtype, device=base_mask_resized.device)
            OBJECT_MASK[:, :, :new_obj_w] = object_mask_resized
            BASE_MASK[:, :, new_obj_w:] = base_mask_resized

        elif layout == 'top-bottom':
            if custom_size > 0:
                new_base_w = custom_size
                new_base_h = int(base_h * (custom_size / base_w))
                base_image = base_image.movedim(-1, 1)
                base_image = common_upscale(base_image, new_base_w, new_base_h, 'bicubic', 'disabled')
                base_image = base_image.movedim(1, -1)
                if base_mask is not None:
                    base_mask = upscale_mask(base_mask, new_base_w, new_base_h)
                base_h, base_w = new_base_h, new_base_w

            scale = base_w / obj_w
            new_obj_h = int(obj_h * scale)
            object_image = object_image.movedim(-1, 1)
            object_image = common_upscale(object_image, base_w, new_obj_h, 'bicubic', 'disabled')
            object_image = object_image.movedim(1, -1)
            if object_mask is not None:
                object_mask = upscale_mask(object_mask, base_w, new_obj_h)
            else:
                object_mask = torch.full((1, new_obj_h, base_w), 1, dtype=torch.float32, device="cpu")

            if object_image.shape[-1] != base_image.shape[-1]:
                min_c = min(object_image.shape[-1], base_image.shape[-1])
                object_image = object_image[..., :min_c]
                base_image = base_image[..., :min_c]
            
            image = torch.cat((object_image, base_image), dim=1)
            batch = object_mask.shape[0]
            out_h = new_obj_h + base_h
            out_w = base_w
            object_mask_resized = object_mask
            base_mask_resized = base_mask
            
            if object_mask_resized.shape[-2:] != (new_obj_h, base_w):
                object_mask_resized = upscale_mask(object_mask_resized, base_w, new_obj_h)
            if base_mask_resized.shape[-2:] != (base_h, base_w):
                base_mask_resized = upscale_mask(base_mask_resized, base_w, base_h)
            
            OBJECT_MASK = torch.zeros((batch, out_h, out_w), dtype=object_mask_resized.dtype, device=object_mask_resized.device)
            BASE_MASK = torch.zeros((batch, out_h, out_w), dtype=base_mask_resized.dtype, device=base_mask_resized.device)
            OBJECT_MASK[:, :new_obj_h, :] = object_mask_resized
            BASE_MASK[:, new_obj_h:, :] = base_mask_resized

        x = object_image.shape[2] if layout == 'left-right' else 0
        y = object_image.shape[1] if layout == 'top-bottom' else 0

        return (image, OBJECT_MASK, BASE_MASK, out_w, out_h, x, y)
    
            
# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AILab_LoadImage": AILab_LoadImage,
    "AILab_Preview": AILab_Preview,
    "AILab_ImagePreview": AILab_ImagePreview,
    "AILab_MaskPreview": AILab_MaskPreview,
    "AILab_ImageMaskConvert": AILab_ImageMaskConvert,
    "AILab_MaskEnhancer": AILab_MaskEnhancer,
    "AILab_MaskCombiner": AILab_MaskCombiner,
    "AILab_ImageCombiner": AILab_ImageCombiner,
    "AILab_MaskExtractor": AILab_MaskExtractor,
    "AILab_ImageStitch": AILab_ImageStitch,
    "AILab_ImageCrop": AILab_ImageCrop,
    "AILab_ICLoRAConcat": AILab_ICLoRAConcat,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_LoadImage": "Load Image (RMBG) 🖼️",
    "AILab_Preview": "Preview (RMBG) 🖼️🎭",
    "AILab_ImagePreview": "Image Preview (RMBG) 🖼️",
    "AILab_MaskPreview": "Mask Preview (RMBG) 🎭",
    "AILab_ImageMaskConvert": "Image/Mask Converter (RMBG) 🖼️🎭",
    "AILab_MaskEnhancer": "Mask Enhancer (RMBG) 🎭",
    "AILab_MaskCombiner": "Mask Combiner (RMBG) 🎭",
    "AILab_ImageCombiner": "Image Combiner (RMBG) 🖼️",
    "AILab_MaskExtractor": "Mask Extractor (RMBG) 🎭",
    "AILab_ImageStitch": "Image Stitch (RMBG) 🖼️",
    "AILab_ImageCrop": "Image Crop (RMBG) 🖼️",
    "AILab_ICLoRAConcat": "IC LoRA Concat (RMBG) 🖼️🎭",
}