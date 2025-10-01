# ml/scripts/run_pipeline.py
"""
End-to-end change detection pipeline:
AOI + Time Range → Satellite Images → Model Inference → Human Report
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
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = img1, img2
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Enhance difference detection
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    
    # Apply threshold with lower sensitivity
    threshold_value = 15  # Reduced from 20 for better sensitivity
    _, binary_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Improve noise removal
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return (binary_mask > 0).astype(np.uint8)


def calculate_area_km2(contour, lat_min, lat_max, img_height):
    """Convert pixel area to square kilometers"""
    pixel_area = cv2.contourArea(contour)
    # Calculate km per pixel (approximate at this latitude)
    km_per_deg = 111.32  # at equator
    lat_center = (lat_min + lat_max) / 2
    km_per_deg_at_lat = km_per_deg * np.cos(np.radians(lat_center))
    deg_per_pixel = (lat_max - lat_min) / img_height
    km_per_pixel = deg_per_pixel * km_per_deg_at_lat
    return pixel_area * (km_per_pixel ** 2)


def generate_report(bin_mask, t0_path, t1_path, start_date, end_date, lat_min, lat_max):
    """Generate detailed change detection report with area calculations"""
    total_pixels = bin_mask.size
    changed_pixels = np.sum(bin_mask > 0)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Find contours
    contours, _ = cv2.findContours(
        (bin_mask * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Calculate areas with minimum size threshold
    significant_regions = [c for c in contours if cv2.contourArea(c) > 100]  # Increased threshold
    total_changed_area_px = sum(cv2.contourArea(c) for c in significant_regions)
    total_changed_area_km2 = sum(
        calculate_area_km2(c, lat_min, lat_max, bin_mask.shape[0]) 
        for c in significant_regions
    )
    
    # More accurate change percentage
    change_percentage = (total_changed_area_px / total_pixels) * 100
    
    # Build report
    report = []
    report.append("📊 CHANGE DETECTION REPORT")
    report.append("-" * 60)
    report.append(f"📍 Location Analysis:")
    report.append(f"   Coordinates: {lat_min:.4f}°N to {lat_max:.4f}°N")
    report.append("")
    report.append(f"📅 Time Period:")
    report.append(f"   T0 (Before): {start_date}")
    report.append(f"   T1 (After):  {end_date}")
    report.append("")
    report.append(f"📈 Change Statistics:")
    report.append(f"   Changed Area:         {total_changed_area_km2:.2f} km²")
    report.append(f"   Change Percentage:    {change_percentage:.2f}%")
    report.append(f"   Changed Regions:      {len(significant_regions)}")
    report.append("")
    
    # Interpretation
    if change_percentage < 1.0:
        status = "✅ MINIMAL CHANGE"
        interpretation = "Very little change detected. Area appears stable."
        color = "🟢"
    elif change_percentage < 5.0:
        status = "⚠️ MINOR CHANGE"
        interpretation = "Small changes detected. May indicate minor development or seasonal variation."
        color = "🟡"
    elif change_percentage < 15.0:
        status = "🟡 MODERATE CHANGE"
        interpretation = "Moderate changes detected. Likely indicates active development or land use changes."
        color = "🟠"
    else:
        status = "🔴 SIGNIFICANT CHANGE"
        interpretation = "Major changes detected. Indicates substantial urban development or transformation."
        color = "🔴"
    
    report.append(f"🎯 Assessment: {status}")
    report.append(f"   {interpretation}")
    report.append("")
    report.append(f"💾 Outputs saved to: outputs/")
    report.append(f"   • change_map.png - Visual comparison")
    report.append(f"   • images/t0.png - Before image")
    report.append(f"   • images/t1.png - After image")
    
    # Save report to file
    report_text = "\n".join(report)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with UTF-8 encoding
    with open(output_dir / "report.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


def save_visualization(bin_mask, t0_path, t1_path):
    """Create and save change visualization"""
    # Load and enhance images
    img1 = cv2.imread(str(t0_path))
    img2 = cv2.imread(str(t1_path))
    
    # Apply image enhancement
    def enhance_image(img):
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl,a,b))
        
        # Convert back to BGR
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    img1 = enhance_image(img1)
    img2 = enhance_image(img2)
    
    # Ensure same size
    h = min(img1.shape[0], img2.shape[0], bin_mask.shape[0])
    w = min(img1.shape[1], img2.shape[1], bin_mask.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    bin_mask_resized = bin_mask[:h, :w]
    
    # Create change overlay
    change_overlay = img2.copy()
    change_overlay[bin_mask_resized > 0] = [255, 0, 0]  # Red for changes
    
    # Blend
    result = cv2.addWeighted(img2, 0.6, change_overlay, 0.4, 0)
    
    # Create side-by-side comparison
    canvas = np.zeros((h, w*3, 3), dtype=np.uint8)
    canvas[:, :w] = img1
    canvas[:, w:2*w] = img2
    canvas[:, 2*w:] = result
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'T0 (Before)', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, 'T1 (After)', (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, 'Changes (Red)', (2*w+10, 30), font, 1, (255, 255, 255), 2)
    
    # Save
    output_path = Path("outputs/change_map.png")
    Image.fromarray(canvas).save(output_path)
    print(f"✅ Visualization saved: {output_path}")


def main(args=None):
    print("🚀 UrbanEyeML – Change Detection Pipeline")
    print("="*60)

    # Use command line args if no args provided
    if args is None:
        args = sys.argv
        
    # Check for environment variables
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("\n❌ ERROR: .env file not found!")
        print(f"Expected location: {env_path}")
        print("\n📝 Please create .env file with:")
        print("   SH_CLIENT_ID=your_client_id")
        print("   SH_CLIENT_SECRET=your_client_secret")
        print("   SH_INSTANCE_ID=your_instance_id")
        print("\n🔗 Get credentials from:")
        print("   https://shapps.dataspace.copernicus.eu/dashboard/")
        return

    # Load environment variables
    load_dotenv(env_path)
    
    if not all([os.getenv('SH_CLIENT_ID'), 
                os.getenv('SH_CLIENT_SECRET'),
                os.getenv('SH_INSTANCE_ID')]):
        print("\n❌ ERROR: Missing required environment variables!")
        print("Please check your .env file has all required credentials")
        return

    try:
        lat_min = float(args[1])
        lon_min = float(args[2])
        lat_max = float(args[3])
        lon_max = float(args[4])
        start_date = args[5]
        end_date = args[6]
    except (ValueError, IndexError) as e:
        print(f"❌ Invalid arguments: {e}")
        return

    # Validate AOI
    if lat_min >= lat_max or lon_min >= lon_max:
        print("❌ Invalid AOI: min values must be less than max values")
        return

    aoi = [lon_min, lat_min, lon_max, lat_max]
    
    # Create output directory
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

    # Step 1: Fetch T0 image
    print(f"\n📍 Fetching T0 ({start_date})...")
    print(f"   AOI: {aoi}")
    img_t0 = fetch_sentinel_image(aoi, start_date, str(t0_path))
    
    if img_t0 is None or img_t0.mean() < 1:
        print("\n❌ Failed to fetch valid T0 image")
        print("💡 Troubleshooting:")
        print("   • Try a different date range")
        print("   • Check for cloud coverage")
        print("   • Verify coordinates are correct")
        return

    # Step 2: Fetch T1 image
    print(f"\n📍 Fetching T1 ({end_date})...")
    img_t1 = fetch_sentinel_image(aoi, end_date, str(t1_path))
    
    if img_t1 is None or img_t1.mean() < 1:
        print("\n❌ Failed to fetch valid T1 image")
        print("💡 Troubleshooting:")
        print("   • Try a different end date")
        print("   • Check for cloud coverage")
        return

    if np.array_equal(img_t0, img_t1):
        print("\n⚠️ Warning: Images are identical!")
        print("💡 Try different dates or coordinates")
        return

    print(f"\n✅ Successfully fetched both images")

    # Step 3: Preprocess
    print("\n🛠️ Preprocessing images...")
    t0 = preprocess_image(img_t0)
    t1 = preprocess_image(img_t1)

    # Step 4: Run inference
    print("🤖 Running change detection model...")
    bin_mask = run_inference(t0, t1)

    # Step 5: Generate report
    print("📝 Generating report...")
    report = generate_report(bin_mask, str(t0_path), str(t1_path), 
                           start_date, end_date, lat_min, lat_max)

    # Step 6: Save visualization
    print("🎨 Creating visualization...")
    save_visualization(bin_mask, str(t0_path), str(t1_path))

    # Recommended dates for best image quality
    RECOMMENDED_DATES = {
        "winter": ("2024-01-15", "2024-01-30"),  # Clear skies
        "summer": ("2023-05-15", "2023-05-30"),  # Less cloud cover
        "spring": ("2024-03-15", "2024-03-30")   # Good visibility
    }
    
    if start_date not in [d[0] for d in RECOMMENDED_DATES.values()]:
        print("\n💡 Tip: For best image quality, try these date ranges:")
        for season, (start, end) in RECOMMENDED_DATES.items():
            print(f"   • {season.title()}: {start} to {end}")
    
    # Display results
    print("\n" + "="*60)
    print(report)
    print("="*60)


if __name__ == "__main__":
    main()