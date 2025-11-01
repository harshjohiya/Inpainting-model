"""
Comprehensive Stable Diffusion Inpainting for Inpaint-Anything
Replaces LaMa with semantic, structure-aware diffusion models
Complete drop-in replacement with auto-prompting and optimization
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional, Union, List
import gc
import warnings


class SDInpainter:
    """
    Production-ready Stable Diffusion Inpainting wrapper
    Designed as drop-in replacement for LaMa in Inpaint-Anything
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda",
        use_fp16: bool = True,
        optimize_memory: bool = True
    ):
        """
        Initialize SD Inpainting pipeline
        
        Args:
            model_id: Model to use:
                - "stabilityai/stable-diffusion-2-inpainting" (RECOMMENDED - best quality)
                - "runwayml/stable-diffusion-inpainting" (SD 1.5 - faster)
            device: "cuda" or "cpu"
            use_fp16: Half precision (faster, less VRAM)
            optimize_memory: Enable memory optimizations
        """
        self.device = device
        self.model_id = model_id
        self.pipe = None
        self.use_fp16 = use_fp16
        self.optimize_memory = optimize_memory
        
        print(f"\n{'='*60}")
        print(f"Initializing Stable Diffusion Inpainting")
        print(f"Model: {model_id}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
    
    def _load_pipeline(self):
        """Lazy load pipeline (only when first needed)"""
        if self.pipe is not None:
            return
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError:
            raise ImportError(
                "\n❌ diffusers library not installed!\n\n"
                "Install with:\n"
                "pip install diffusers transformers accelerate\n"
            )
        
        print("Loading model... (this takes ~30 seconds on first run)")
        
        # Load with appropriate dtype
        if self.use_fp16 and self.device == "cuda":
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                safety_checker=None,
                requires_safety_checker=False
            )
        
        self.pipe = self.pipe.to(self.device)
        
        # Memory optimizations
        if self.optimize_memory and self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
                print("✓ Attention slicing enabled (saves VRAM)")
            except:
                pass
            
            try:
                self.pipe.enable_vae_slicing()
                print("✓ VAE slicing enabled")
            except:
                pass
            
            # Try to enable xformers if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xformers enabled (faster inference)")
            except:
                pass
        
        print("✓ Model loaded successfully!\n")
    
    def inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.99,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Main inpainting function - drop-in replacement for lama_inpaint
        
        Args:
            img: RGB numpy array (H, W, 3), values 0-255
            mask: Binary mask (H, W), values 0 or 255
                  255 = inpaint region, 0 = keep original
            prompt: Text description of what to generate
                   If None, auto-generates based on scene analysis
            negative_prompt: What to avoid (use default if None)
            num_inference_steps: Quality vs speed (20=fast, 50=best quality)
            guidance_scale: Prompt adherence (7.5 default, higher = more strict)
            strength: How much to change (0.99 = full inpainting)
            seed: For reproducible results
            
        Returns:
            Inpainted image as numpy array (H, W, 3), values 0-255
        """
        # Ensure pipeline is loaded
        self._load_pipeline()
        
        # Convert numpy to PIL
        img_pil = Image.fromarray(img.astype(np.uint8))
        
        # Process mask
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask)
        
        # Auto-generate prompt if not provided
        if prompt is None:
            prompt = self._auto_prompt(img, mask)
            print(f"🤖 Auto-prompt: '{prompt}'")
        
        # Default negative prompt for quality
        if negative_prompt is None:
            negative_prompt = (
                "blurry, distorted, low quality, artifacts, watermark, "
                "text, duplicate, morbid, mutilated, poorly drawn, "
                "extra limbs, missing limbs, jpeg artifacts, ugly"
            )
        
        # Resize to SD-friendly dimensions (multiples of 64)
        original_size = img_pil.size
        target_size = self._optimal_size(original_size)
        
        needs_resize = (target_size != original_size)
        if needs_resize:
            img_pil_resized = img_pil.resize(target_size, Image.LANCZOS)
            mask_pil_resized = mask_pil.resize(target_size, Image.NEAREST)
            print(f"📐 Resized: {original_size} → {target_size}")
        else:
            img_pil_resized = img_pil
            mask_pil_resized = mask_pil
        
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run inpainting
        print(f"🎨 Inpainting with {num_inference_steps} steps...")
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=img_pil_resized,
                mask_image=mask_pil_resized,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                negative_prompt=negative_prompt
            ).images[0]
        
        # Resize back if needed
        if needs_resize:
            result = result.resize(original_size, Image.LANCZOS)
        
        # Convert back to numpy
        result_np = np.array(result)
        
        # Cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        print("✓ Inpainting complete!\n")
        return result_np
    
    def _optimal_size(self, size: tuple) -> tuple:
        """Round to nearest multiple of 64 (SD requirement)"""
        w, h = size
        w = ((w + 31) // 64) * 64
        h = ((h + 31) // 64) * 64
        
        # Limit to 1024 to avoid VRAM issues
        max_dim = 1024
        if w > max_dim or h > max_dim:
            scale = min(max_dim / w, max_dim / h)
            w = int(w * scale)
            h = int(h * scale)
            w = (w // 64) * 64
            h = (h // 64) * 64
        
        return (w, h)
    
    def _auto_prompt(self, img: np.ndarray, mask: np.ndarray) -> str:
        """
        Intelligently generate prompt based on scene analysis
        """
        h, w = img.shape[:2]
        mask_binary = (mask > 127)
        
        # Analyze mask position
        mask_coords = np.argwhere(mask_binary)
        if len(mask_coords) == 0:
            return "a clean indoor scene, photorealistic, high quality"
        
        avg_y = np.mean(mask_coords[:, 0])
        y_ratio = avg_y / h
        
        # Analyze surrounding colors
        unmasked = img[~mask_binary]
        if len(unmasked) > 100:
            brightness = np.mean(unmasked)
            
            # Floor region (bottom 40%)
            if y_ratio > 0.6:
                if brightness > 180:
                    return "a clean white marble floor, bright indoor lighting, photorealistic, high quality, detailed"
                else:
                    return "a clean floor with tiles, indoor lighting, photorealistic, high quality, detailed"
            
            # Wall region (top 40%)
            elif y_ratio < 0.4:
                if brightness > 180:
                    return "a clean white wall, bright indoor room, photorealistic, high quality, detailed"
                else:
                    return "a clean painted wall, indoor room, photorealistic, high quality, detailed"
            
            # Middle region (furniture area)
            else:
                if brightness > 180:
                    return "a clean white wall and marble floor, bright indoor room, photorealistic, high quality, detailed"
                else:
                    return "a clean wall and floor, empty indoor room, photorealistic, high quality, detailed"
        
        return "a clean indoor room, photorealistic, high quality, detailed"


# =============================================================================
# Global singleton instance
# =============================================================================
_global_sd_inpainter = None

def get_sd_inpainter(device="cuda") -> SDInpainter:
    """Get or create global SD inpainter instance"""
    global _global_sd_inpainter
    if _global_sd_inpainter is None:
        _global_sd_inpainter = SDInpainter(device=device)
    return _global_sd_inpainter


# =============================================================================
# Drop-in replacement functions (API-compatible with lama_inpaint.py)
# =============================================================================

def inpaint_img_with_sd(
    img: np.ndarray,
    mask: np.ndarray,
    prompt: Optional[str] = None,
    device: str = "cuda",
    steps: int = 30
) -> np.ndarray:
    """
    Drop-in replacement for inpaint_img_with_lama()
    
    Args:
        img: RGB image (H, W, 3), 0-255
        mask: Binary mask (H, W), 255=inpaint
        prompt: Optional text prompt
        device: "cuda" or "cpu"
        steps: Inference steps (20-50)
        
    Returns:
        Inpainted image (H, W, 3), 0-255
    """
    inpainter = get_sd_inpainter(device)
    return inpainter.inpaint(
        img, mask,
        prompt=prompt,
        num_inference_steps=steps
    )


# Alias for compatibility
inpaint_img_with_sd_auto = inpaint_img_with_sd


if __name__ == "__main__":
    print("""
    Stable Diffusion Inpainting Integration
    ========================================
    
    Usage Example:
    
    ```python
    from sd_inpaint_integrated import inpaint_img_with_sd
    from utils import load_img_to_array
    
    # Load image and mask
    img = load_img_to_array("image.jpg")
    mask = load_img_to_array("mask.png")
    
    # Option 1: Auto prompt (recommended)
    result = inpaint_img_with_sd(img, mask, device="cuda")
    
    # Option 2: Custom prompt
    result = inpaint_img_with_sd(
        img, mask,
        prompt="a clean white wall and marble floor",
        device="cuda",
        steps=30
    )
    
    # Save result
    from PIL import Image
    Image.fromarray(result).save("result.jpg")
    ```
    
    Installation:
    pip install diffusers transformers accelerate
    
    For best performance, also install:
    pip install xformers  # Much faster
    """)
