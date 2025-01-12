# ComfyUI-RMBG v1.5.0
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, and BEN. It leverages deep learning techniques
# to process images and generate masks for background removal.

# Models License Notice:
# - RMBG-2.0: Apache-2.0 License (https://huggingface.co/briaai/RMBG-2.0)
# - INSPYRENET: MIT License (https://github.com/plemeri/InSPyReNet)
# - BEN: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN)
# 
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import folder_paths
from PIL import ImageFilter
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import shutil
import sys
import importlib.util
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

# Add model path
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# Model configuration
AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "briaai/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py"
        },
        "cache_dir": "RMBG-2.0"
    },
    "INSPYRENET": {
        "type": "inspyrenet",
        "repo_id": "1038lab/inspyrenet",
        "files": {
            "inspyrenet.safetensors": "inspyrenet.safetensors"
        },
        "cache_dir": "INSPYRENET"
    },
    "BEN": {
        "type": "ben",
        "repo_id": "PramaLLC/BEN",
        "files": {
            "model.py": "model.py",
            "BEN_Base.pth": "BEN_Base.pth"
        },
        "cache_dir": "BEN"
    }
}

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)

class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
    def get_cache_dir(self, model_name):
        return os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])
    
    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.current_model_version = None
            torch.cuda.empty_cache()
            print("Model cleared from memory")

class RMBGModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            self.model = AutoModelForImageSegmentation.from_pretrained(
                cache_dir,
                trust_remote_code=True,
                local_files_only=True
            )
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
            
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)

            # Prepare batch processing
            transform_image = transforms.Compose([
                transforms.Resize((params["process_res"], params["process_res"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Ensure input is in list format
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            # Store original image sizes
            original_sizes = [tensor2pil(img).size for img in images]

            # Batch process transformations
            input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in images]
            input_batch = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                results = self.model(input_batch)[-1].sigmoid().cpu()
                masks = []
                
                # Process each result and resize back to original dimensions
                for i, (result, (orig_w, orig_h)) in enumerate(zip(results, original_sizes)):
                    result = result.squeeze()
                    result = result * (1 + (1 - params["sensitivity"]))
                    result = torch.clamp(result, 0, 1)
                    
                    # Resize back to original dimensions
                    result = F.interpolate(result.unsqueeze(0).unsqueeze(0),
                                         size=(orig_h, orig_w),
                                         mode='bilinear').squeeze()
                    
                    masks.append(tensor2pil(result))

                return masks

        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")

class InspyrenetModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                import transparent_background
                self.model = transparent_background.Remover()
                self.current_model_version = model_name
            except ImportError:
                try:
                    import pip
                    pip.main(['install', 'transparent_background'])
                    import transparent_background
                    self.model = transparent_background.Remover()
                    self.current_model_version = model_name
                except Exception as e:
                    handle_model_error(f"Failed to install transparent_background: {str(e)}")
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            # Resize for processing
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            # Process image
            foreground = self.model.process(resized_image, type='rgba')
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in Inspyrenet processing: {str(e)}")

class BENModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            model_path = os.path.join(cache_dir, "model.py")
            module_name = f"custom_ben_model_{hash(model_path)}"
            
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            ben_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = ben_module
            spec.loader.exec_module(ben_module)
            
            model_weights_path = os.path.join(cache_dir, "BEN_Base.pth")
            self.model = ben_module.BEN_Base()
            self.model.loadcheckpoints(model_weights_path)
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            processed_input = resized_image.convert("RGBA")
            
            with torch.no_grad():
                _, foreground = self.model.inference(processed_input)
            
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in BEN processing: {str(e)}")

class RMBG:
    def __init__(self):
        self.models = {
            "RMBG-2.0": RMBGModel(),
            "INSPYRENET": InspyrenetModel(),
            "BEN": BENModel()
        }
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image to be processed for background removal.",
            "model": "Select the background removal model to use (RMBG-2.0, INSPYRENET, BEN).",
            "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
            "process_res": "Set the processing resolution (higher values require more VRAM and may increase processing time).",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "background": "Choose the background color for the final output (Alpha for transparent background).",
            "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
            "optimize": "Enable model optimization for faster processing (may affect output quality)."
        }
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (list(AVAILABLE_MODELS.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "background": (["Alpha", "black", "white", "gray", "green", "blue", "red"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "optimize": (["default", "on"], {"default": "default", "tooltip": tooltips["optimize"]})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "🧪AILab/🧽RMBG"

    def process_image(self, image, model, **params):
        try:
            processed_images = []
            processed_masks = []
            
            bg_colors = {
                "Alpha": None,
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "gray": (128, 128, 128),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "red": (255, 0, 0)
            }
            
            model_instance = self.models[model]
            
            # Check and download model if needed
            cache_status, message = model_instance.check_model_cache(model)
            if not cache_status:
                print(f"Cache check: {message}")
                print("Downloading required model files...")
                download_status, download_message = model_instance.download_model(model)
                if not download_status:
                    handle_model_error(download_message)
                print("Model files downloaded successfully")
            
            for img in image:
                # Get mask from specific model
                mask = model_instance.process_image(img, model, params)
                
                # Ensure mask is in the correct format
                if isinstance(mask, list):
                    masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
                    mask = masks[0] if masks else None
                elif isinstance(mask, Image.Image):
                    mask = mask.convert("L")

                # Post-process mask
                mask_tensor = pil2tensor(mask)
                mask_tensor = mask_tensor * (1 + (1 - params["sensitivity"]))
                mask_tensor = torch.clamp(mask_tensor, 0, 1)
                mask = tensor2pil(mask_tensor)
                
                if params["mask_blur"] > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))
                
                if params["mask_offset"] != 0:
                    if params["mask_offset"] > 0:
                        for _ in range(params["mask_offset"]):
                            mask = mask.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-params["mask_offset"]):
                            mask = mask.filter(ImageFilter.MinFilter(3))
                
                if params["invert_output"]:
                    mask = Image.fromarray(255 - np.array(mask))

                # Create final image
                orig_image = tensor2pil(img)
                orig_rgba = orig_image.convert("RGBA")
                r, g, b, _ = orig_rgba.split()
                foreground = Image.merge('RGBA', (r, g, b, mask))

                if params["background"] != "Alpha":
                    bg_color = bg_colors[params["background"]]
                    bg_image = Image.new('RGBA', orig_image.size, (*bg_color, 255))
                    composite_image = Image.alpha_composite(bg_image, foreground)
                    
                    # Convert to RGB if background is not Alpha
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                else:
                    # Keep as RGBA if background is Alpha
                    processed_images.append(pil2tensor(foreground))
                
                processed_masks.append(pil2tensor(mask))

            return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0))
            
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "RMBG": RMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RMBG": "Remove Background (RMBG)"
} 