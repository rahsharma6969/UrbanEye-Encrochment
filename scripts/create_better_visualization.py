"""
IMPROVED Visualization - Makes changes HIGHLY VISIBLE
"""

import cv2
import numpy as np
from pathlib import Path


def create_high_contrast_visualization(bin_mask, t0_path, t1_path, start_date, end_date, classification_results=None):
    """
    Create ULTRA HIGH CONTRAST visualization
    Red changes are BRIGHT and OBVIOUS
    """
    
    print("\n🎨 Creating HIGH CONTRAST visualization...")
    
    # Load images
    img1 = cv2.imread(str(t0_path))
    img2 = cv2.imread(str(t1_path))
    
    if img1 is None or img2 is None:
        print("❌ Failed to load images")
        return
    
    # Get dimensions
    h, w = bin_mask.shape[:2]
    
    # UPSCALE 4X for better visibility
    scale_factor = 4
    img1 = cv2.resize(img1, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_CUBIC)
    bin_mask = cv2.resize(bin_mask.astype(np.uint8), (w*scale_factor, h*scale_factor), interpolation=cv2.INTER_NEAREST)
    
    h, w = h*scale_factor, w*scale_factor
    
    # ENHANCED CONTRAST on images
    img1 = enhance_contrast(img1)
    img2 = enhance_contrast(img2)
    
    # Create BRIGHT RED overlay for changes
    overlay = img2.copy()
    
    # Make red SUPER BRIGHT
    red_mask = (bin_mask > 0)
    overlay[red_mask] = [0, 0, 255]  # Pure red
    
    # Add yellow border around changes for extra visibility
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(bin_mask, kernel, iterations=2)
    border_mask = (dilated_mask > 0) & (bin_mask == 0)
    overlay[border_mask] = [0, 255, 255]  # Yellow border
    
    # Blend with HIGH alpha for visibility
    changes_vis = cv2.addWeighted(img2, 0.5, overlay, 0.5, 0)
    
    # Create side-by-side with LARGE header
    header_height = 150
    canvas = np.zeros((h + header_height, w * 3, 3), dtype=np.uint8)
    
    # Dark gradient header
    for i in range(header_height):
        intensity = int(30 + 20 * (1 - i/header_height))
        canvas[i, :] = [intensity, intensity, intensity]
    
    # Place images
    canvas[header_height:, 0:w] = img1
    canvas[header_height:, w:2*w] = img2
    canvas[header_height:, 2*w:3*w] = changes_vis
    
    # Format dates
    try:
        from datetime import datetime
        t0_obj = datetime.strptime(start_date, "%Y-%m-%d")
        t1_obj = datetime.strptime(end_date, "%Y-%m-%d")
        t0_display = t0_obj.strftime("%b %d, %Y")
        t1_display = t1_obj.strftime("%b %d, %Y")
    except:
        t0_display = start_date
        t1_display = end_date
    
    # LARGE, BOLD text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_large_text(img, text, x, y, scale, color=(255, 255, 255)):
        # Black shadow
        cv2.putText(img, text, (x+3, y+3), font, scale, (0, 0, 0), 6, cv2.LINE_AA)
        # White text
        cv2.putText(img, text, (x, y), font, scale, color, 3, cv2.LINE_AA)
    
    # Headers
    draw_large_text(canvas, 'T0 (BEFORE)', 20, 50, 1.2)
    draw_large_text(canvas, t0_display, 20, 100, 0.8, (200, 200, 200))
    
    draw_large_text(canvas, 'T1 (AFTER)', w + 20, 50, 1.2)
    draw_large_text(canvas, t1_display, w + 20, 100, 0.8, (200, 200, 200))
    
    draw_large_text(canvas, 'CHANGES DETECTED', 2*w + 20, 50, 1.2, (255, 255, 0))
    draw_large_text(canvas, 'Red = Changes', 2*w + 20, 100, 0.8, (0, 0, 255))
    
    # Add classification if available
    if classification_results:
        classification = classification_results['classification']
        severity = classification_results['severity']
        confidence = classification_results['confidence']
        
        # Color based on severity
        severity_colors = {
            'HIGH': (0, 0, 255),      # Red
            'MEDIUM': (0, 165, 255),  # Orange
            'LOW': (0, 255, 0)        # Green
        }
        color = severity_colors.get(severity, (255, 255, 255))
        
        # Classification text
        class_text = f"{classification} ({confidence*100:.0f}%)"
        draw_large_text(canvas, class_text, w + 20, 135, 0.7, color)
    
    # Calculate and show change percentage PROMINENTLY
    change_pct = (np.sum(bin_mask > 0) / bin_mask.size) * 100
    change_text = f"Change: {change_pct:.2f}%"
    draw_large_text(canvas, change_text, 2*w + 20, 135, 0.7, (255, 255, 0))
    
    # Save
    output_path = Path("outputs/high_contrast_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    
    print(f"✅ High-contrast visualization saved: {output_path}")
    print(f"📊 Change detected: {change_pct:.2f}% of image")
    print(f"🔴 Changed pixels: {np.sum(bin_mask > 0):,}")
    
    return canvas


def enhance_contrast(img):
    """Enhance image contrast using CLAHE"""
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Increase saturation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced


if __name__ == "__main__":
    # Test with existing data
    from pathlib import Path
    import json
    
    raw_dir = Path("data/raw")
    t0_path = raw_dir / "t0.png"
    t1_path = raw_dir / "t1.png"
    
    if not t0_path.exists():
        print("❌ No images found. Run pipeline first.")
        exit(1)
    
    # Load change mask
    from run_pipeline import preprocess_image, run_inference
    
    img_t0 = cv2.imread(str(t0_path))
    img_t1 = cv2.imread(str(t1_path))
    
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    
    bin_mask = run_inference(t0_norm, t1_norm)
    
    # Load classification results if available
    classification_file = Path("outputs/classification_results.json")
    classification_results = None
    if classification_file.exists():
        with open(classification_file, 'r') as f:
            classification_results = json.load(f)
    
    # Create visualization
    create_high_contrast_visualization(
        bin_mask,
        str(t0_path),
        str(t1_path),
        "2023-06-01",
        "2024-06-01",
        classification_results
    )
    
    print("\n✅ Done! Check: outputs/high_contrast_comparison.png")
