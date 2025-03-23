# ComfyUI-RMBG Version Compatibility Guide

## Transformers Version Compatibility

The ComfyUI-RMBG node uses the Hugging Face transformers library to load and run the RMBG-2.0 model. Starting from version 2.1.1, we have added compatibility support for newer versions of transformers while maintaining compatibility with older versions.

### Recommended Versions

**Recommended transformers version range**: `4.35.0 - 4.48.3`

These versions have been thoroughly tested and are fully compatible with the RMBG-2.0 model. If you do not have specific requirements, it is advisable to use versions within this range.

### Compatibility with Higher Versions

From version 2.1.1 onwards, ComfyUI-RMBG supports higher versions of transformers (such as 4.49.0+). This resolves conflicts with other models that require higher versions of transformers (e.g., hunyuanvideo).

When a higher version of transformers is detected, ComfyUI-RMBG will automatically switch to compatibility mode, attempting to load and process model files directly without relying on APIs that may have changed in higher versions.

### Troubleshooting

If you encounter issues while using ComfyUI-RMBG, here are some common problems and solutions:

#### Issue: `'Config' object has no attribute 'get_text_config'` error

**Cause**: This typically occurs when using transformers version 4.49.0 or higher, where the API has changed.

**Solution**:
1. **Upgrade ComfyUI-RMBG**: Ensure you are using version 2.1.1 or higher, which includes compatibility fixes.
2. **Downgrade transformers**: If upgrading ComfyUI-RMBG does not resolve the issue, you can try downgrading transformers:
   ```
   pip uninstall transformers
   pip install transformers==4.48.3
   ```

#### Issue: Conflicts when using other models that require higher versions of transformers

**Cause**: Some models (like hunyuanvideo) require transformers version 4.49.0 or higher.

**Solution**:
1. Upgrade to ComfyUI-RMBG version 2.1.1 or higher, which supports running under higher versions of transformers.
2. If issues persist, consider running these models in different environments or using virtual environments to isolate dependencies.

#### Issue: Other errors when loading models or processing images

**Solution**:
1. Check the error messages to understand the specific issue.
2. Ensure that the RMBG-2.0 model files are correctly downloaded to the `ComfyUI/models/RMBG/RMBG-2.0` directory.
3. Try re-downloading the model files.
4. If the problem persists, please submit an issue on GitHub with the complete error message.

## Environment Management Recommendations

If you need to use multiple models with different transformers version requirements simultaneously, consider the following methods:

1. **Use virtual environments**: Create separate virtual environments for different models, installing specific versions of dependencies in each environment.
2. **Use Docker**: Run different versions of ComfyUI and models in Docker containers.
3. **Prioritize compatibility**: Use models and tools that have broader compatibility whenever possible.

## Changelog

Please refer to [update.md](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md) for detailed updates on each version.