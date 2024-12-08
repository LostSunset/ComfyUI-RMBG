# ComfyUI-RMBG

A ComfyUI custom node designed for advanced image background removal utilizing multiple models, including RMBG-2.0, INSPYRENET, and BEN.

$${\color{red}If\ this\ custom\ node\ helps\ you\ or\ you\ like\ my\ work,\ please\ give\ me⭐on\ this\ repo!}$$
$${\color{red}It's\ a\ greatest\ encouragement\ for\ my\ efforts!}$$

## News & Updates
- 2024/12/02: Update Comfyui-RMBG ComfyUI Custom Node to v1.2.1 ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v121-20241202) )
![GIF_TO_AWEBP](https://github.com/user-attachments/assets/7f8275d5-06e5-4880-adfe-930f045df673)

- 2024/11/29: Update Comfyui-RMBG ComfyUI Custom Node to v1.2.0 ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v120-20241129) )
![RMBGv1 2 0](https://github.com/user-attachments/assets/4fd10123-6c95-4f9e-8d25-fdb39b5fc792)

- 2024/11/21: Update Comfyui-RMBG ComfyUI Custom Node to v1.1.0 ( [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v110-20241121) )
![comfyui-rmbg version compare](https://github.com/user-attachments/assets/2d23cf42-ca74-49e5-a8bf-9de377bd71aa)

## Features

![RMBG Demo](https://github.com/user-attachments/assets/f3ffa3c4-5a21-4c0c-a078-b4ffe681c4c4)

## Installation

1. install on ComfyUI-Manager, search `Comfyui-RMBG` and install.
![image](https://github.com/user-attachments/assets/419db32c-3e52-4276-bc83-7782363e0aa0)
   Install the dependencies `requirements.txt` file within the ComfyUI-RMBG folder.
   ```bash
   ./ComfyUI/python_embeded/python -m pip install -r requirements.txt
   ```

3. Clone this repository to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG
```

3. Manually download the models:
- The model will be automatically downloaded to `ComfyUI/models/RMBG/` when first time using the custom node.
- Manually download the RMBG-2.0 model by visiting this [link](https://huggingface.co/briaai/RMBG-2.0/tree/main), then download the files and place them in the `/ComfyUI/models/RMBG/RMBG-2.0` folder.
- Manually download the INSPYRENET models by visiting the [link](https://huggingface.co/1038lab/inspyrenet), then download the files and place them in the `/ComfyUI/models/INSPYRENET` folder.
- Manually download the BEN model by visiting the [link](https://huggingface.co/PramaLLC/BEN), then download the files and place them in the `/ComfyUI/models/BEN` folder.

## Usage
![RMBG](https://github.com/user-attachments/assets/cd0eb92e-8f2e-4ae4-95f1-899a6d83cab6)

### Optional Settings :bulb: Tips
| Optional Settings    | :memo: Description                                                           | :bulb: Tips                                                                                   |
|----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Sensitivity**      | Adjusts the strength of mask detection. Higher values result in stricter detection. | Default value is 0.5. Adjust based on image complexity; more complex images may require higher sensitivity. |
| **Processing Resolution** | Controls the processing resolution of the input image, affecting detail and memory usage. | Choose a value between 256 and 2048, with a default of 1024. Higher resolutions provide better detail but increase memory consumption. |
| **Mask Blur**        | Controls the amount of blur applied to the mask edges, reducing jaggedness. | Default value is 0. Try setting it between 1 and 5 for smoother edge effects.                    |
| **Mask Offset**      | Allows for expanding or shrinking the mask boundary. Positive values expand the boundary, while negative values shrink it. | Default value is 0. Adjust based on the specific image, typically fine-tuning between -10 and 10. |
| **Background**      | Choose output background color | Alpha (transparent background) Black, White, Green, Blue, Red |
| **Invert Output**      | Flip mask and image output | Invert both image and mask output |
| **Performance Optimization** | Properly setting options can enhance performance when processing multiple images. | If memory allows, consider increasing `process_res` and `mask_blur` values for better results, but be mindful of memory usage. |

### Basic Usage
1. Load `RMBG (Remove Background)` node from the `🧪AILab/🧽RMBG` category
2. Connect an image to the input
3. Select a model from the dropdown menu
4. select the parameters as needed (optional)
3. Get two outputs:
   - IMAGE: Processed image with transparent, black, white, green, blue, or red background
   - MASK: Binary mask of the foreground

### Parameters
- `sensitivity`: Controls the background removal sensitivity (0.0-1.0)
- `process_res`: Processing resolution (512-2048, step 128)
- `mask_blur`: Blur amount for the mask (0-64)
- `mask_offset`: Adjust mask edges (-20 to 20)
- `background`: Choose output background color
- `invert_output`: Flip mask and image output
- `optimize`: Toggle model optimization

<details>
<summary><h2>About Models</h2></summary>

## RMBG-2.0
RMBG-2.0 is is developed by BRIA AI and uses the BiRefNet architecture which includes:
- High accuracy in complex environments
- Precise edge detection and preservation
- Excellent handling of fine details
- Support for multiple objects in a single image
- Output Comparison
- Output with background
- Batch output for video
The model is trained on a diverse dataset of over 15,000 high-quality images, ensuring:
- Balanced representation across different image types
- High accuracy in various scenarios
- Robust performance with complex backgrounds

## INSPYRENET
INSPYRENET is specialized in human portrait segmentation, offering:
- Fast processing speed
- Good edge detection capability
- Ideal for portrait photos and human subjects

## BEN
BEN is robust on various image types, offering:
- Good balance between speed and accuracy
- Effective on both simple and complex scenes
- Suitable for batch processing
</details>


## Requirements
- ComfyUI
- Python 3.10+
- Required packages (automatically installed):
  - torch>=2.0.0
  - torchvision>=0.15.0
  - Pillow>=9.0.0
  - numpy>=1.22.0
  - huggingface-hub>=0.19.0
  - tqdm>=4.65.0
  - transformers>=4.35.0
  - transparent-background>=1.2.4

## Credits
- RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
- INSPYRENET: https://github.com/plemeri/InSPyReNet
- BEN: https://huggingface.co/PramaLLC/BEN
- Created by: [1038 Lab](https://github.com/1038lab)

## License
GPL-3.0 License
