# ComfyUI-RMBG Update Log

## v1.4.0 (2025/1/2)

### New Clothes Segment Node
- Added intelligent clothes segmentation functionality
  - Support for 18 different clothing categories (Hat, Hair, Face, Sunglasses, Upper-clothes, etc.)
  - Multiple item selection for combined segmentation
  - Same parameter controls as other RMBG nodes (process_res, mask_blur, mask_offset, background options)
  - Automatic model downloading and resource management

![rmbg_v1 4 0](https://github.com/user-attachments/assets/978c168b-03a8-4937-aa03-06385f34b820)

## v1.3.2 (2024/12/29)

### Updates
- Enhanced background handling to support RGBA output when "Alpha" is selected.
- Ensured RGB output for all other background color selections.

## v1.3.1 (2024/12/25)

### Bug Fixes
- Fixed an issue with mask processing when the model returns a list of masks.
- Improved handling of image formats to prevent processing errors.

## v1.3.0 (2024/12/23)

### New Segment (RMBG) Node
- Text-Prompted Intelligent Object Segmentation
  - Use natural language prompts (e.g., "a cat", "red car") to identify and segment target objects
  - Support for multiple object detection and segmentation
  - Perfect for precise object extraction and recognition tasks

![rmbg v1.3.0](https://github.com/user-attachments/assets/7607546e-ffcb-45e2-ab90-83267292757e)

### Supported Models
- SAM (Segment Anything Model)
  - sam_vit_h: 2.56GB - Highest accuracy
  - sam_vit_l: 1.25GB - Balanced performance
  - sam_vit_b: 375MB - Lightweight option
- GroundingDINO
  - SwinT: 694MB - Fast and efficient
  - SwinB: 938MB - Higher precision

### Key Features
- Intuitive Parameter Controls
  - Threshold: Adjust detection precision
  - Mask Blur: Smooth edges
  - Mask Offset: Expand or shrink selection
  - Background Options: Alpha/Black/White/Green/Blue/Red
- Automatic Model Management
  - Auto-download models on first use
  - Smart GPU memory handling

### Usage Examples
1. Tag-Style Prompts
   - Single object: "cat"
   - Multiple objects: "cat, dog, person"
   - With attributes: "red car, blue shirt"
   - Format: Use commas to separate multiple objects (e.g., "a, b, c")

2. Natural Language Prompts
   - Simple sentence: "a person wearing a red jacket"
   - Complex scene: "a woman in a blue dress standing next to a car"
   - With location: "a cat sitting on the sofa"
   - Format: Write a natural descriptive sentence

3. Tips for Better Results
   - For Tag Style:
     - Separate objects with commas: "chair, table, lamp"
     - Add attributes before objects: "wooden chair, glass table"
     - Keep it simple and clear
   - For Natural Language:
     - Use complete sentences
     - Include details like color, position, action
     - Be as descriptive as needed
   - Parameter Adjustments:
     - Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
     - Use mask blur for smoother edges
     - Adjust mask offset to fine-tune selection

## v1.2.2 (2024/12/12)
![RMBG1 2 2](https://github.com/user-attachments/assets/cb7b1ad0-a2ca-4369-9401-54957af6c636)

### Improvements
- Changed INSPYRENET model format from .pth to .safetensors for:
  - Better security
  - Faster loading speed (2-3x faster)
  - Improved memory efficiency
  - Better cross-platform compatibility
- Simplified node display name for better UI integration

## v1.2.1 (2024/12/02)

### New Features
- ANPG (animated PNG), AWEBP (animated WebP) and GIF supported.

https://github.com/user-attachments/assets/40ec0b27-4fa2-4c99-9aea-5afad9ca62a5

### Bug Fixes
- Fixed video processing issue

### Performance Improvements
- Enhanced batch processing in RMBG-2.0 model
- Added support for proper batch image handling
- Improved memory efficiency by optimizing image size handling

### Technical Details
- Added original size preservation for maintaining aspect ratios
- Implemented proper batch tensor processing
- Improved error handling and code robustness
- Performance gains:
  - Single image processing: ~5-10% improvement
  - Batch processing: up to 30-50% improvement (depending on batch size and GPU)

## v1.2.0 (2024/11/29)

### Major Changes
- Combined three background removal models into one unified node
- Added support for RMBG-2.0, INSPYRENET, and BEN models
- Implemented lazy loading for models (only downloads when first used)

### Model Introduction
- RMBG-2.0 ([Homepage](https://huggingface.co/briaai/RMBG-2.0))
  - Latest version of RMBG model
  - Excellent performance on complex backgrounds
  - High accuracy in preserving fine details
  - Best for general purpose background removal

- INSPYRENET ([Homepage](https://github.com/plemeri/InSPyReNet))
  - Specialized in human portrait segmentation
  - Fast processing speed
  - Good edge detection capability
  - Ideal for portrait photos and human subjects

- BEN (Background Elimination Network) ([Homepage](https://huggingface.co/PramaLLC/BEN))
  - Robust performance on various image types
  - Good balance between speed and accuracy
  - Effective on both simple and complex scenes
  - Suitable for batch processing

### Features
- Unified interface for all three models
- Common parameters for all models:
  - Sensitivity adjustment
  - Processing resolution control
  - Mask blur and offset options
  - Multiple background color options
  - Invert output option
  - Model optimization toggle

### Improvements
- Optimized memory usage with model clearing
- Enhanced error handling and user feedback
- Added detailed tooltips for all parameters
- Improved mask post-processing

### Dependencies
- Updated all package dependencies to latest stable versions
- Added support for transparent-background package
- Optimized dependency management

## v1.1.0 (2024/11/21)

### New Features
- Added background color options
  - Alpha (transparent background)
  - Black, White, Green, Blue, Red

![RMBG_v1 1 0](https://github.com/user-attachments/assets/b7cbadff-5386-4d96-bc34-a19ad34efb4b)

- Improved mask processing
  - Better detail preservation
  - Enhanced edge quality
  - More accurate segmentation
    
![rmbg version compare](https://github.com/user-attachments/assets/8339aa8e-46db-4f11-aa7b-0a710f0a1711)

- Added video batch processing
  - Support for video file background removal
  - Maintains original video framerate and resolution
  - Multiple output format support (with Alpha channel)
  - Efficient batch processing for video frames

https://github.com/user-attachments/assets/259220d3-c148-4030-93d6-c17dd5bccee1

- Added model cache management
  - Cache status checking
  - Model memory cleanup
  - Better error handling

### Parameter Updates
- Renamed 'invert_mask' to 'invert_output' for clarity
- Added sensitivity adjustment for mask strength
- Updated tooltips for better clarity

### Technical Improvements
- Optimized image processing pipeline
- Added proper model cache verification
- Improved memory management
- Better error handling and recovery
- Enhanced batch processing performance for videos

### Dependencies
- Added timm>=0.6.12,<1.0.0 for model support
- Updated requirements.txt with version constraints

### Bug Fixes
- Fixed mask detail preservation issues
- Improved mask edge quality
- Fixed memory leaks in model handling

### Usage Notes
- The 'Alpha' background option provides transparent background
- Sensitivity parameter now controls mask strength
- Model cache is checked before each operation
- Memory is automatically cleaned when switching models
- Video processing supports various formats and maintains quality
