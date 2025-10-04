# ml/scripts/run_pipeline.py
"""
UrbanEyeML – End-to-end Change Detection Pipeline
AOI + Time Range → Satellite Images → Your Trained Model → Human Report
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import cv2
from dotenv import load_dotenv
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Load .env file
load_dotenv()
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    print("❌ ERROR: .env file not found!")
    print(f"Create it at: {env_path}")
    print("Content:")
    print("SH_CLIENT_ID=your-client-id")
    print("SH_CLIENT_SECRET=your-client-secret")
    exit(1)

def preprocess_image(img):
    """Normalize image to [0,1] float32"""
    return img.astype(np.float32) / 255.0

# Global cache for model
_model_cache = None

@torch.no_grad()
def run_inference(t0, t1, threshold=0.5, min_area=100):
    """
    Use your trained UNet model
    Input: t0, t1 — (H,W,3), normalized [0,1]
    Output: bin_mask — (H,W), uint8 {0,1}, change_probs — (H,W) float [0,1]
    """
    global _model_cache

    if _model_cache is None:
        print("🧠 Loading your trained UNet model...")
        try:
            # Initialize model
            model = smp.Unet(
                encoder_name="resnet34",
                in_channels=6,
                classes=2,
                encoder_weights=None
            ).to("cpu")

            ckpt_path = "outputs/checkpoints/best.pth"
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"Model checkpoint missing: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get('model_state', ckpt)

            new_state = {}
            for k, v in state.items():
                nk = k[len("module."):] if k.startswith("module.") else k
                new_state[nk] = v

            model.load_state_dict(new_state)
            model.eval()
            _model_cache = model

            print("✅ Loaded trained model from best.pth")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    # Stack T0 and T1 into 6-channel input
    x = np.concatenate([t0, t1], axis=-1)  # (H,W,6)
    x = np.transpose(x, (2, 0, 1))         # (6,H,W)
    x = torch.from_numpy(x).unsqueeze(0).float()  # (1,6,H,W)

    # Forward pass
    logits = _model_cache(x)
    probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # (H,W)

    # Use the provided threshold and min_area
    bin_mask = (probs >= threshold).astype(np.uint8)

    # Remove small components
    try:
        from scipy import ndimage
        labels, count = ndimage.label(bin_mask)
        out = np.zeros_like(bin_mask)
        for i in range(1, count + 1):
            area = (labels == i).sum()
            if area >= min_area:
                out[labels == i] = 1
        bin_mask = out
    except ImportError:
        pass

    return bin_mask, probs

def calculate_area_km2(contour, lat_min, lat_max, img_height):
    """Convert pixel area to square kilometers"""
    pixel_area = cv2.contourArea(contour)
    km_per_deg = 111.32
    lat_center = (lat_min + lat_max) / 2
    km_per_deg_at_lat = km_per_deg * np.cos(np.radians(lat_center))
    deg_per_pixel = (lat_max - lat_min) / img_height
    km_per_pixel = deg_per_pixel * km_per_deg_at_lat
    return pixel_area * (km_per_pixel ** 2)

def generate_human_readable_report(bin_mask, t0_path, t1_path, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """Generate a simple, human-readable report for non-technical users"""
    total_pixels = bin_mask.size
    changed_pixels = bin_mask.sum()
    change_percentage = (changed_pixels / total_pixels) * 100

    try:
        contours, _ = cv2.findContours(
            (bin_mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        significant_regions = [c for c in contours if cv2.contourArea(c) > 50]
        total_changed_area_km2 = sum(
            calculate_area_km2(c, lat_min, lat_max, bin_mask.shape[0])
            for c in significant_regions
        )
    except Exception:
        significant_regions = []
        total_changed_area_km2 = 0.0

    # Determine location description
    location_name = "Mumbai Metropolitan Area"
    if 19.10 <= lat_min <= 19.30 and 72.80 <= lon_min <= 73.00:
        location_name = "Mumbai, India"
    elif 19.18 <= lat_min <= 19.22 and 72.97 <= lon_min <= 73.02:
        location_name = "Thane, India"
    
    # Calculate time span in months
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months_diff = (end.year - start.year) * 12 + (end.month - start.month)
    
    # Create a simple, conversational report
    report = [
        "🌍 URBAN CHANGE REPORT",
        "=" * 60,
        "",
        f"📍 LOCATION: {location_name}",
        f"📅 TIME PERIOD: {start_date} to {end_date} (about {months_diff} months)",
        "",
        "📊 WHAT WE FOUND:",
        ""
    ]
    
    # Explain changes in simple terms
    if change_percentage < 0.5:
        report.append("   • Very few changes detected in this area")
        report.append("   • The area appears to have remained mostly the same")
        report.append("   • Any changes are likely minor or seasonal")
    elif change_percentage < 2.0:
        report.append("   • Some changes were detected in the area")
        report.append("   • These could be new buildings, roads, or other developments")
        report.append("   • The changes affect a small portion of the area")
    elif change_percentage < 5.0:
        report.append("   • Moderate changes were detected in the area")
        report.append("   • There appears to be ongoing development or construction")
        report.append("   • These changes are visible in satellite images")
    else:
        report.append("   • Significant changes were detected in the area")
        report.append("   • This suggests major development or construction activity")
        report.append("   • The landscape of this area has changed noticeably")
    
    report.append("")
    report.append("📈 BY THE NUMBERS:")
    report.append(f"   • Total area changed: {total_changed_area_km2:.2f} square kilometers")
    report.append(f"   • Percentage of area with changes: {change_percentage:.2f}%")
    report.append(f"   • Number of distinct change spots: {len(significant_regions)}")
    
    # Add relatable comparisons
    if total_changed_area_km2 > 0:
        if total_changed_area_km2 < 0.5:
            report.append(f"   • This is about the size of {int(total_changed_area_km2 * 200)} football fields")
        elif total_changed_area_km2 < 2.0:
            report.append(f"   • This is about the size of {int(total_changed_area_km2 * 50)} football fields")
        else:
            report.append(f"   • This is a significant area, roughly {total_changed_area_km2:.1f} square kilometers")
    
    report.append("")
    report.append("🎯 WHAT THIS MEANS:")
    
    # Provide interpretation based on change percentage
    if change_percentage < 0.5:
        report.append("   • This area is stable with little development activity")
        report.append("   • Any changes are likely routine maintenance or seasonal")
        report.append("   • No major construction or development detected")
    elif change_percentage < 2.0:
        report.append("   • There is some development activity in this area")
        report.append("   • Changes could include new buildings, roads, or land use")
        report.append("   • The development appears to be limited in scope")
    elif change_percentage < 5.0:
        report.append("   • This area is experiencing active development")
        report.append("   • Multiple construction projects may be underway")
        report.append("   • The landscape is being modified by human activity")
    else:
        report.append("   • This area is undergoing major transformation")
        report.append("   • Large-scale development is changing the landscape")
        report.append("   • This could be new infrastructure, housing, or commercial projects")
    
    report.append("")
    report.append("📋 FILES CREATED:")
    report.append("   • change_map.png - Shows before/after images with changes highlighted in red")
    report.append("   • change_heatmap.png - Black and white map showing where changes occurred")
    report.append("   • images/t0.png - Satellite image from the start date")
    report.append("   • images/t1.png - Satellite image from the end date")
    report.append("")
    report.append("💡 HOW TO USE THIS INFORMATION:")
    report.append("   • Look at change_map.png to see exactly where changes happened")
    report.append("   • Red areas in the images show where development occurred")
    report.append("   • The heatmap helps identify the most changed areas")
    report.append("   • Compare the before and after images to see the transformation")
    report.append("")
    report.append("=" * 60)
    report.append("This report was generated by UrbanEyeML - an AI system that")
    report.append("analyzes satellite images to detect changes in urban areas.")
    report.append("For questions or more detailed analysis, please contact the team.")
    
    return "\n".join(report)


def generate_bw_heatmap(bin_mask, output_path):
    """Generate a black and white heatmap highlighting changed areas"""
    try:
        # Create a black background
        heatmap = np.zeros((bin_mask.shape[0], bin_mask.shape[1]), dtype=np.uint8)
        
        # Set changed areas to white (255)
        heatmap[bin_mask > 0] = 255
        
        # Apply morphological operations to make changes more visible
        kernel = np.ones((3, 3), np.uint8)
        heatmap = cv2.dilate(heatmap, kernel, iterations=1)
        
        # Save the heatmap
        Image.fromarray(heatmap).save(output_path)
        print(f"🌑 Saved black and white heatmap: {output_path}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to generate black and white heatmap: {e}")
        return False

def generate_grayscale_heatmap(change_probs, output_path):
    """Generate a grayscale heatmap based on change probability"""
    try:
        # Convert probability to grayscale (0-255)
        grayscale = (change_probs * 255).astype(np.uint8)
        
        # Apply histogram equalization to enhance contrast
        grayscale = cv2.equalizeHist(grayscale)
        
        # Save the grayscale heatmap
        Image.fromarray(grayscale).save(output_path)
        print(f"🌒 Saved grayscale heatmap: {output_path}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to generate grayscale heatmap: {e}")
        return False

def enhance_image_quality(img):
    """Apply professional-grade image enhancement using CLAHE"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    except Exception as e:
        print(f"⚠️ Enhancement failed, using original: {e}")
        return img

def save_visualization(bin_mask, t0_path, t1_path, change_probs=None):
    """Create HIGH-QUALITY visualization with BOLD RED highlights and clear dividers"""
    try:
        print("🎨 Creating enhanced visualization...")
        
        # Load images using OpenCV for better quality control
        t0 = cv2.imread(str(t0_path))
        t1 = cv2.imread(str(t1_path))
        
        if t0 is None or t1 is None:
            print("❌ Failed to load images")
            return False
        
        # Enhance image quality
        print("   → Enhancing image quality...")
        t0 = enhance_image_quality(t0)
        t1 = enhance_image_quality(t1)
        
        # Ensure same dimensions
        h = min(t0.shape[0], t1.shape[0], bin_mask.shape[0])
        w = min(t0.shape[1], t1.shape[1], bin_mask.shape[1])
        t0 = t0[:h, :w]
        t1 = t1[:h, :w]
        
        # Resize binary mask to match image dimensions
        if bin_mask.shape[0] != h or bin_mask.shape[1] != w:
            bin_mask_resized = cv2.resize(
                bin_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            bin_mask_resized = bin_mask.astype(np.uint8)
        
        # Check for changes
        change_pixels = np.sum(bin_mask_resized > 0)
        change_percentage = (change_pixels / (w * h)) * 100
        print(f"   → Detected changes: {change_pixels} pixels ({change_percentage:.2f}%)")
        
        # Fallback: Use probability map if no changes detected
        if change_pixels < 10 and change_probs is not None:
            print("   → Using probability map (low detection)...")
            probs_resized = cv2.resize(change_probs, (w, h))
            bin_mask_resized = (probs_resized >= 0.15).astype(np.uint8)
            change_pixels = np.sum(bin_mask_resized > 0)
            print(f"   → After adjustment: {change_pixels} pixels ({change_pixels/(w*h)*100:.2f}%)")
        
        # AGGRESSIVE dilation to make changes ULTRA VISIBLE
        kernel = np.ones((5, 5), np.uint8)
        bin_mask_dilated = cv2.dilate(bin_mask_resized, kernel, iterations=2)
        
        # Create PURE RED overlay in BGR format
        print("   → Creating red overlay...")
        change_overlay = t1.copy()
        change_overlay[bin_mask_dilated > 0] = [0, 0, 255]  # PURE RED in BGR
        
        # STRONG blending (50% original + 50% red for MAXIMUM VISIBILITY)
        result = cv2.addWeighted(t1, 0.5, change_overlay, 0.5, 0)
        
        # Add THICK WHITE borders around changes
        contours, _ = cv2.findContours(bin_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 3)
        
        print(f"   → Highlighted {len(contours)} change regions")
        
        # Create location description from coordinates
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        location_desc = f"Lat: {lat_center:.4f}°N, Lon: {lon_center:.4f}°E"
        
        # Create dividers (white vertical lines)
        divider_width = 5
        divider_color = (255, 255, 255)  # White
        
        # Create canvas with extra space for dividers and labels
        canvas_width = w * 3 + divider_width * 2
        canvas_height = h + 60  # Extra space for labels at top and bottom
        
        # Create black canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Position variables
        x0 = 0
        x1 = w
        x2 = w + divider_width
        x3 = w + divider_width + w
        x4 = w + divider_width + w + divider_width
        
        y_start = 40  # Leave space for top labels
        
        # Place images
        canvas[y_start:y_start+h, x0:x1] = t0
        canvas[y_start:y_start+h, x2:x3] = t1
        canvas[y_start:y_start+h, x4:x4+w] = result
        
        # Add vertical dividers
        canvas[:, x1:x1+divider_width] = divider_color
        canvas[:, x3:x3+divider_width] = divider_color
        
        # Add horizontal divider at top
        canvas[35:40, :] = divider_color
        
        # Add horizontal divider at bottom
        canvas[y_start+h:y_start+h+5, :] = divider_color
        
        # Add location label at top center
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for centering
        (text_w, text_h), _ = cv2.getTextSize(location_desc, font, font_scale, thickness)
        text_x = (canvas_width - text_w) // 2
        cv2.putText(canvas, location_desc, (text_x, 25), font, font_scale, (255, 255, 255), thickness)
        
        # Add image labels at bottom
        labels = [
            ('BEFORE', x0 + w//2 - 40),
            ('AFTER', x2 + w//2 - 35),
            ('CHANGES', x4 + w//2 - 45)
        ]
        
        for text, x_center in labels:
            # Get text size for centering
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x_pos = x_center - text_w // 2
            y_pos = y_start + h + 20
            cv2.putText(canvas, text, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        # Add date labels below image labels
        date_font_scale = 0.6
        date_labels = [
            (start_date, x0 + w//2 - 40),
            (end_date, x2 + w//2 - 35),
            ('HIGHLIGHTED', x4 + w//2 - 60)
        ]
        
        for text, x_center in date_labels:
            (text_w, text_h), _ = cv2.getTextSize(text, font, date_font_scale, thickness)
            x_pos = x_center - text_w // 2
            y_pos = y_start + h + 45
            cv2.putText(canvas, text, (x_pos, y_pos), font, date_font_scale, (200, 200, 200), thickness)
        
        # Save with MAXIMUM quality (zero compression)
        out_path = Path("outputs") / "change_map.png"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Use zero compression for perfect quality
        cv2.imwrite(
            str(out_path),
            canvas,
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
        
        print(f"✅ Saved HIGH-QUALITY visualization: {out_path}")
        print(f"   Resolution: {canvas.shape[1]}x{canvas.shape[0]} pixels")
        print(f"   Quality: Maximum (zero compression)")
        print(f"   Change regions: {len(contours)}")
        print(f"   Location: {location_desc}")
        
        return True

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def main():
    print("🚀 UrbanEyeML – AI-Powered Change Detection Pipeline")
    print("="*60)

    # Check .env file
    if not env_path.exists():
        print("❌ ERROR: .env file not found!")
        return

    # Parse arguments
    if len(sys.argv) < 7:
        print("Usage:")
        print("  python -m scripts.run_pipeline <lat_min> <lon_min> <lat_max> <lon_max> <start_date> <end_date> [threshold] [min_area]")
        print("\nExample:")
        print("  python -m scripts.run_pipeline 19.10 72.80 19.30 73.00 2023-02-01 2023-11-30 0.3 50")
        print("\n💡 Tips:")
        print("  • Use dates 6+ months apart for visible changes")
        print("  • Use summer dates (Feb-Nov) to avoid monsoon clouds")
        print("  • AOI should be small (< 0.2° × 0.2°)")
        print("  • Lower threshold = more sensitive (try 0.3 or 0.2)")
        return

    try:
        lat_min = float(sys.argv[1])
        lon_min = float(sys.argv[2])
        lat_max = float(sys.argv[3])
        lon_max = float(sys.argv[4])
        start_date = sys.argv[5]
        end_date = sys.argv[6]
        
        # Optional parameters - LOWER defaults for better detection
        threshold = 0.3  # More sensitive (was 0.5)
        min_area = 50    # Catch smaller changes (was 100)
        
        if len(sys.argv) >= 8:
            threshold = float(sys.argv[7])
        if len(sys.argv) >= 9:
            min_area = int(sys.argv[8])
        
    except ValueError:
        print("❌ Invalid arguments")
        return

    # Validate AOI
    if lat_min >= lat_max or lon_min >= lon_max:
        print("❌ Invalid AOI: min must be less than max")
        return

    aoi = [lon_min, lat_min, lon_max, lat_max]

    # Output directory
    output_dir = Path("outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0_path = output_dir / "t0.png"
    t1_path = output_dir / "t1.png"

    # Import fetch function
    try:
        from scripts.fetch_sentinel import fetch_sentinel_image
    except ImportError as e:
        print(f"❌ Failed to import fetch_sentinel: {e}")
        return

    # Step 1: Fetch T0
    print(f"\n📍 Fetching T0 ({start_date})...")
    img_t0 = fetch_sentinel_image(aoi, start_date, str(t0_path))
    if img_t0 is None:
        print("❌ Failed to fetch T0 image.")
        print("👉 Make sure:")
        print("   - SH_CLIENT_ID/SECRET are set")
        print("   - URLs point to dataspace.copernicus.eu")
        return

    # Step 2: Fetch T1
    print(f"\n📍 Fetching T1 ({end_date})...")
    img_t1 = fetch_sentinel_image(aoi, end_date, str(t1_path))
    if img_t1 is None:
        print("❌ Failed to fetch T1 image.")
        return

    print(f"\n✅ Successfully fetched both images")

    # Step 3: Preprocess
    print("🛠️ Preprocessing images...")
    t0 = preprocess_image(img_t0)
    t1 = preprocess_image(img_t1)

    # Step 4: Run inference using YOUR MODEL
    print("🤖 Running YOUR trained model...")
    print(f"   Using threshold: {threshold} (LOWER = more sensitive)")
    print(f"   Using min_area: {min_area}")
    bin_mask, change_probs = run_inference(t0, t1, threshold, min_area)
    
    # Debug: Print statistics about the change mask
    change_percentage = np.sum(bin_mask) / bin_mask.size * 100
    print(f"📊 Change detection statistics:")
    print(f"   - Threshold: {threshold}")
    print(f"   - Min area: {min_area}")
    print(f"   - Change percentage: {change_percentage:.2f}%")
    print(f"   - Max probability: {np.max(change_probs):.2f}")
    print(f"   - Mean probability: {np.mean(change_probs):.2f}")

    # Step 5: Generate black and white heatmap
    print("\n🌑 Generating black and white heatmap...")
    heatmap_path = Path("outputs") / "change_heatmap.png"
    generate_bw_heatmap(bin_mask, heatmap_path)
    
    # Step 6: Generate grayscale heatmap based on probability
    print("🌒 Generating grayscale heatmap...")
    grayscale_path = Path("outputs") / "grayscale_heatmap.png"
    generate_grayscale_heatmap(change_probs, grayscale_path)

    # Step 7: Generate report
    print("\n📝 Generating report...")
    report = generate_human_readable_report(bin_mask, str(t0_path), str(t1_path), start_date, end_date, lat_min, lat_max, lon_min, lon_max)
    
    # Step 8: Save ENHANCED visualization with BOLD red highlights
    print("\n🎨 Creating ULTRA HIGH-QUALITY visualization...")
    save_visualization(bin_mask, str(t0_path), str(t1_path), change_probs)

    # Final output
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print("\n🎉 SUCCESS! Check outputs/change_map.png for BOLD RED highlights!")
    print("💡 Tip: If changes still not visible, try lower threshold (0.2) or different dates")

if __name__ == "__main__":
    main()


'''
# Standard usage (more sensitive)
python -m scripts.run_pipeline 19.10 72.80 19.30 73.00 2023-02-01 2023-11-30

# Even more sensitive
python -m scripts.run_pipeline 19.10 72.80 19.30 73.00 2023-02-01 2023-11-30 0.2 20

# Custom parameters
python -m scripts.run_pipeline 19.10 72.80 19.30 73.00 2023-02-01 2023-11-30 0.25 30
'''
