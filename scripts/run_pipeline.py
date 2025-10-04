# ml/scripts/run_pipeline.py
"""
End-to-end change detection pipeline:
AOI + Time Range → Satellite Images → Model Inference → Human Report
✅ Added image enhancement
✅ Added actual dates to visualization
✅ Fixed UTF-8 encoding for report
✅ Added image validation
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv

# Add ml/ to path
sys.path.append(str(Path(__file__).parent.parent))
# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))


def preprocess_image(img):
    """Normalize image to 0-1 range"""
    if img.max() > 1.0:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def run_inference(t0, t1):
    """Run change detection inference"""
    # Convert to uint8 for OpenCV
    img1 = (t0 * 255).astype(np.uint8) if t0.max() <= 1.0 else t0.astype(np.uint8)
    img2 = (t1 * 255).astype(np.uint8) if t1.max() <= 1.0 else t1.astype(np.uint8)
    
    # Ensure same size
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2
    
    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return (binary > 0).astype(np.uint8)


def save_visualization(bin_mask, t0_path, t1_path, start_date, end_date):
    """
    Create and save change visualization with VISIBLE dates
    """
    # Load images
    img1 = cv2.imread(str(t0_path))
    img2 = cv2.imread(str(t1_path))
    
    if img1 is None or img2 is None:
        print("❌ Failed to load images for visualization")
        return
    
    # Ensure same size
    h = min(img1.shape[0], img2.shape[0], bin_mask.shape[0])
    w = min(img1.shape[1], img2.shape[1], bin_mask.shape[1])
    
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    bin_mask_resized = cv2.resize(bin_mask.astype(np.uint8), (w, h))
    
    # Create red overlay for changes
    red_overlay = np.zeros_like(img2)
    red_overlay[:, :, 2] = bin_mask_resized * 255  # Red channel
    
    # Blend with T1 image
    result = cv2.addWeighted(img2, 0.7, red_overlay, 0.3, 0)
    
    # Create side-by-side comparison with BLACK BACKGROUND at top for text
    header_height = 80  # Space for text
    canvas = np.zeros((h + header_height, w * 3, 3), dtype=np.uint8)
    
    # Place images
    canvas[header_height:, :w] = img1
    canvas[header_height:, w:2*w] = img2
    canvas[header_height:, 2*w:] = result
    
    # Format dates
    try:
        from datetime import datetime
        t0_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        t1_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        t0_display = t0_date_obj.strftime("%b %d, %Y")
        t1_display = t1_date_obj.strftime("%b %d, %Y")
        date_range = f'{t0_date_obj.strftime("%b %d")} to {t1_date_obj.strftime("%b %d, %Y")}'
    except Exception:
        t0_display = start_date
        t1_display = end_date
        date_range = f'{start_date} to {end_date}'
    
    # ✨ Draw text with black outline for visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_text_with_outline(img, text, position, font_scale, thickness):
        x, y = position
        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)   # outline
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)  # fill
    
    # T0 labels
    draw_text_with_outline(canvas, 'T0 (Before)', (10, 30), 0.9, 2)
    draw_text_with_outline(canvas, t0_display, (10, 60), 0.6, 1)
    
    # T1 labels
    draw_text_with_outline(canvas, 'T1 (After)', (w + 10, 30), 0.9, 2)
    draw_text_with_outline(canvas, t1_display, (w + 10, 60), 0.6, 1)
    
    # Changes labels
    draw_text_with_outline(canvas, 'Changes (Red)', (2 * w + 10, 30), 0.9, 2)
    draw_text_with_outline(canvas, date_range, (2 * w + 10, 60), 0.5, 1)
    
    # Save
    output_path = Path("outputs/change_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"✅ Visualization saved: {output_path}")


def main(args=None):
    """Main pipeline execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run change detection pipeline with enhancement")
    parser.add_argument("lat_min", type=float, help="Minimum latitude")
    parser.add_argument("lon_min", type=float, help="Minimum longitude")
    parser.add_argument("lat_max", type=float, help="Maximum latitude")
    parser.add_argument("lon_max", type=float, help="Maximum longitude")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable image enhancement")
    
    args = parser.parse_args(args)
    
    # Setup paths
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    t0_path = raw_dir / "t0.png"
    t1_path = raw_dir / "t1.png"
    
    # Define AOI
    aoi = [args.lon_min, args.lat_min, args.lon_max, args.lat_max]
    start_date = args.start_date
    end_date = args.end_date
    
    print("=" * 60)
    print("🛰️  URBAN CHANGE DETECTION PIPELINE WITH ENHANCEMENT")
    print("=" * 60)
    print(f"📍 AOI: {aoi}")
    print(f"📅 Time Range: {start_date} → {end_date}")
    print(f"🎨 Image Enhancement: {'Enabled' if not args.no_enhance else 'Disabled'}")
    print("=" * 60)
    
    # Import fetch function
    from fetch_sentinel import fetch_sentinel_image
    
    # Step 1: Fetch T0 image with enhancement
    print(f"\n{'='*60}")
    print(f"📍 FETCHING T0 IMAGE ({start_date})")
    print(f"{'='*60}")
    img_t0, actual_start_date = fetch_sentinel_image(
        aoi, start_date, str(t0_path), enhance=not args.no_enhance
    )
    
    if img_t0 is None:
        print("\n❌ CRITICAL: Failed to fetch valid T0 image")
        print("   Possible solutions:")
        print("   1. Try different dates with better satellite coverage")
        print("   2. Check Sentinel Hub credentials in .env file")
        print("   3. Try a different AOI (area of interest)")
        print("   4. Check internet connection")
        return
    
    if img_t0.mean() < 1 or img_t0.mean() > 250:
        print(f"\n❌ CRITICAL: T0 image is invalid (mean: {img_t0.mean():.2f})")
        print("   The image appears to be blank or corrupted")
        return
    
    # Update with actual date if available
    if actual_start_date:
        start_date = actual_start_date
        print(f"✅ Using actual T0 date: {start_date}")
    
    # Step 2: Fetch T1 image with enhancement
    print(f"\n{'='*60}")
    print(f"📍 FETCHING T1 IMAGE ({end_date})")
    print(f"{'='*60}")
    img_t1, actual_end_date = fetch_sentinel_image(
        aoi, end_date, str(t1_path), enhance=not args.no_enhance
    )
    
    if img_t1 is None:
        print("\n❌ CRITICAL: Failed to fetch valid T1 image")
        print("   Possible solutions:")
        print("   1. Try different dates with better satellite coverage")
        print("   2. Increase time window in fetch_sentinel.py")
        return
    
    if img_t1.mean() < 1 or img_t1.mean() > 250:
        print(f"\n❌ CRITICAL: T1 image is invalid (mean: {img_t1.mean():.2f})")
        print("   The image appears to be blank or corrupted")
        return
    
    # Update with actual date if available
    if actual_end_date:
        end_date = actual_end_date
        print(f"✅ Using actual T1 date: {end_date}")
    
    # Step 3: Preprocess
    print("\n🔄 Preprocessing images...")
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    print("✅ Preprocessing complete")
    
    # Step 4: Run inference
    print("\n🤖 Running change detection...")
    bin_mask = run_inference(t0_norm, t1_norm)
    print("✅ Change detection complete")
    
    # Step 5: Calculate statistics
    total_pixels = bin_mask.size
    changed_pixels = bin_mask.sum()
    pct_change = (changed_pixels / total_pixels) * 100
    
    print(f"\n📊 Results:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Changed pixels: {changed_pixels:,}")
    print(f"  Change percentage: {pct_change:.2f}%")
    
    # Step 6: Save visualization with actual dates
    print("\n🎨 Creating visualization with dates...")
    save_visualization(bin_mask, str(t0_path), str(t1_path), start_date, end_date)
    
    # Step 7: Generate report
    print("\n📝 Generating report...")
    from generate_report import generate_report
    report = generate_report(bin_mask, str(t0_path), str(t1_path))
    
    # ✨ Fix: Add encoding='utf-8' to handle emoji characters
    report_path = output_dir / "report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n📂 Outputs saved to: {output_dir}")
    print(f"  - change_map.png (with dates: {start_date} to {end_date})")
    print(f"  - report.txt")
    print(f"\n📊 Summary:")
    print(f"  - Image quality: Valid (T0 mean: {img_t0.mean():.2f}, T1 mean: {img_t1.mean():.2f})")
    print(f"  - Change detected: {pct_change:.2f}%")


if __name__ == "__main__":
    main()
