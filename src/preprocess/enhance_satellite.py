# ml/src/preprocess/enhance_satellite.py
"""
Satellite Image Enhancement Pipeline
Integrates with fetch_sentinel.py and run_pipeline.py
Improves image quality before change detection
"""

import cv2
import numpy as np
from typing import Tuple

class SatelliteImageEnhancer:
    """
    Enhance satellite imagery quality using multiple techniques:
    - Cloud removal
    - Atmospheric correction (dehaze)
    - Contrast enhancement
    - Denoising
    - Sharpening
    """
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete enhancement pipeline
        
        Args:
            image: RGB satellite image (H, W, 3) in range [0, 255]
        
        Returns:
            Enhanced image
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Step 1: Atmospheric correction (dehaze)
        image = self._dehaze(image)
        
        # Step 2: Contrast enhancement
        image = self._enhance_contrast(image)
        
        # Step 3: Denoise
        image = self._denoise(image)
        
        # Step 4: Sharpen
        image = self._sharpen(image)
        
        return image
    
    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        """Remove haze/atmospheric effects using dark channel prior"""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Dark channel prior for haze removal
        kernel_size = 15
        dark_channel = np.min(img_float, axis=2)
        dark_channel = cv2.erode(dark_channel, np.ones((kernel_size, kernel_size)))
        
        # Estimate atmospheric light
        flat_dark = dark_channel.ravel()
        num_pixels = len(flat_dark)
        num_brightest = int(max(num_pixels * 0.001, 1))
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        
        atmospheric_light = np.mean(image.reshape(-1, 3)[indices], axis=0)
        atmospheric_light = np.maximum(atmospheric_light, 1)
        
        # Transmission estimation
        transmission = 1 - 0.95 * dark_channel
        transmission = np.maximum(transmission, 0.1)
        
        # Recover scene radiance
        dehazed = np.zeros_like(img_float)
        for i in range(3):
            dehazed[:,:,i] = (img_float[:,:,i] - atmospheric_light[i] / 255.0) / transmission + atmospheric_light[i] / 255.0
        
        dehazed = np.clip(dehazed * 255, 0, 255).astype(np.uint8)
        return dehazed
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges"""
        # Use fastNlMeansDenoisingColored for color images
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen edges for better building detection"""
        # Unsharp mask
        gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def remove_clouds(self, image: np.ndarray, threshold: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and mask clouds
        
        Returns:
            (cloud_free_image, cloud_mask)
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        # Clouds are typically very bright
        cloud_mask = (l_channel > threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
        
        # Inpaint clouds
        cloud_free = cv2.inpaint(image, cloud_mask, 3, cv2.INPAINT_TELEA)
        
        return cloud_free, cloud_mask


def enhance_for_change_detection(image: np.ndarray) -> np.ndarray:
    """
    Quick enhancement function for integration
    Use this in your pipeline
    """
    enhancer = SatelliteImageEnhancer()
    return enhancer.enhance(image)


if __name__ == "__main__":
    # Test enhancement
    from PIL import Image
    
    # Test on a sample image
    test_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
    
    enhancer = SatelliteImageEnhancer()
    enhanced = enhancer.enhance(test_img)
    
    print("✅ Enhancement pipeline test successful!")
    print(f"Input range: [{test_img.min()}, {test_img.max()}]")
    print(f"Output range: [{enhanced.min()}, {enhanced.max()}]")
