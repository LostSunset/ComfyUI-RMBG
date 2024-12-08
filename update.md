# ComfyUI-RMBG Update Log

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
    
![rmbg1 Demo](https://github.com/user-attachments/assets/4f7d073c-f9cc-4bdb-875c-ba51decc9d5a)

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
