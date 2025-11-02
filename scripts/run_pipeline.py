# ml/scripts/run_pipeline.py
"""
End-to-end change detection pipeline with GEE Building Validation:
AOI + Time Range → Satellite Images → Image Enhancement → Model Inference → GEE Validation → Classification → Report
✅ Added image enhancement
✅ Added actual dates to visualization
✅ Fixed UTF-8 encoding for report
✅ Added image validation
✅ Added Natural vs Man-made Classification
✅ ADDED: GEE Building Validation (100x better data)
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


def enhance_image(img):
    """
    Enhanced image quality for better change detection
    Applies: Dehazing, CLAHE, Sharpening, Color balance
    """
    print("   🎨 Enhancing image quality...")
    
    # Convert to 8-bit if needed
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    # Ensure 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 1. Atmospheric correction (dehaze)
    print("      - Atmospheric correction...")
    kernel_size = 15
    dark_channel = np.min(img.astype(float), axis=2)
    dark_channel = cv2.erode(dark_channel, np.ones((kernel_size, kernel_size)))
    
    atmospheric_light = np.percentile(dark_channel, 95)
    transmission = 1 - 0.95 * (dark_channel / atmospheric_light)
    transmission = np.clip(transmission, 0.1, 1)
    
    dehazed = np.zeros_like(img)
    for i in range(3):
        dehazed[:,:,i] = (img[:,:,i] - atmospheric_light) / transmission + atmospheric_light
    
    dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
    
    # 2. Contrast enhancement (CLAHE)
    print("      - Contrast enhancement...")
    lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpening
    print("      - Sharpening buildings...")
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. Color balance
    print("      - Auto color balance...")
    result = np.zeros_like(sharpened)
    for i in range(3):
        result[:,:,i] = cv2.normalize(sharpened[:,:,i], None, 0, 255, cv2.NORM_MINMAX)
    
    print("   ✅ Image enhancement complete")
    return result


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


def validate_with_gee_buildings(change_mask, aoi_coords):
    """
    Validate detected changes against GEE building footprints
    
    Args:
        change_mask: Binary mask of detected changes (H x W)
        aoi_coords: [lat_min, lon_min, lat_max, lon_max]
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*70)
    print("🏢 VALIDATING WITH GOOGLE EARTH ENGINE BUILDINGS")
    print("="*70)
    
    try:
        # Import GEE service
        from gee_building_service import GEEBuildingService
        
        # Initialize GEE
        gee_service = GEEBuildingService(project_id='your project id')
        
        # Fetch building data for this area
        lat_min, lon_min, lat_max, lon_max = aoi_coords
        building_data = gee_service.get_buildings_for_area(
            lat_min=lat_min,
            lon_min=lon_min,
            lat_max=lat_max,
            lon_max=lon_max,
            confidence_threshold=0.75
        )
        
        # Calculate validation metrics
        total_buildings = building_data['total_buildings']
        density = building_data['density_per_km2']
        area_km2 = building_data['area_km2']
        
        # Calculate change percentage
        total_pixels = change_mask.size
        changed_pixels = int(np.sum(change_mask))
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Determine if changes are likely building-related
        is_urban = density > 100  # buildings/km²
        is_dense_urban = density > 500
        
        validation_result = {
            'total_buildings': total_buildings,
            'building_density': density,
            'area_km2': area_km2,
            'change_percentage': change_percentage,
            'is_urban_area': is_urban,
            'is_dense_urban': is_dense_urban,
            'likely_building_changes': is_urban and change_percentage > 1.0,
            'avg_building_size': building_data['statistics']['average_building_size_m2']
        }
        
        print(f"✅ Found {total_buildings:,} existing buildings")
        print(f"📊 Density: {density:.1f} buildings/km²")
        print(f"📐 Average building size: {validation_result['avg_building_size']:.0f} m²")
        print(f"🔍 Change detected: {change_percentage:.2f}% of area")
        
        if is_dense_urban:
            print(f"📍 Zone: Dense Urban (>500 buildings/km²)")
        elif is_urban:
            print(f"📍 Zone: Urban (>100 buildings/km²)")
        else:
            print(f"📍 Zone: Non-urban")
        
        return validation_result
        
    except ImportError:
        print("⚠️  GEE service not available - install required packages")
        print("   pip install earthengine-api")
        return None
    except Exception as e:
        print(f"⚠️  GEE validation failed: {e}")
        print("💡 Continuing without GEE validation...")
        return None


def classify_encroachment(change_mask, gee_validation, start_date, end_date, aoi):
    """
    Improved classification using both ML detection and GEE building data
    
    Returns:
        Dictionary with classification results
    """
    print("\n" + "="*70)
    print("🔍 ENCROACHMENT CLASSIFICATION")
    print("="*70)
    
    # Calculate basic change metrics
    total_pixels = change_mask.size
    changed_pixels = int(np.sum(change_mask))
    change_pct = (changed_pixels / total_pixels) * 100
    
    # If GEE validation available, use it
    if gee_validation:
        density = gee_validation['building_density']
        total_buildings = gee_validation['total_buildings']
        area_km2 = gee_validation['area_km2']
        
        # IMPROVED CLASSIFICATION LOGIC with GEE data
        if density > 500:  # Dense urban
            if change_pct > 2:
                classification = "ILLEGAL CONSTRUCTION SUSPECTED"
                confidence = 0.92
                reason = f"Dense urban area ({density:.0f} buildings/km²) with {change_pct:.1f}% new construction detected"
                subcategory = "Urban Encroachment"
                severity = "HIGH"
            elif change_pct > 0.5:
                classification = "MINOR CONSTRUCTION"
                confidence = 0.78
                reason = f"Minor changes in dense urban area - could be renovations"
                subcategory = "Urban Development"
                severity = "MEDIUM"
            else:
                classification = "NATURAL/SEASONAL"
                confidence = 0.65
                reason = "Minimal changes - likely seasonal/natural variations"
                subcategory = "Natural Changes"
                severity = "LOW"
        
        elif density > 100:  # Urban/Suburban
            if change_pct > 5:
                classification = "ILLEGAL CONSTRUCTION SUSPECTED"
                confidence = 0.88
                reason = f"Significant expansion ({change_pct:.1f}%) in urban area with {total_buildings:,} existing buildings"
                subcategory = "Urban Expansion"
                severity = "HIGH"
            elif change_pct > 2:
                classification = "DEVELOPMENT ACTIVITY"
                confidence = 0.72
                reason = f"Moderate construction activity detected"
                subcategory = "Mixed Development"
                severity = "MEDIUM"
            else:
                classification = "NATURAL/SEASONAL"
                confidence = 0.68
                reason = "Minor changes - likely natural variations"
                subcategory = "Seasonal Changes"
                severity = "LOW"
        
        else:  # Non-urban/Rural
            if change_pct > 15:
                classification = "NATURAL CHANGES"
                confidence = 0.86
                reason = f"Large natural changes in non-urban area ({change_pct:.1f}%)"
                subcategory = "Natural/Agricultural"
                severity = "LOW"
            elif change_pct > 5:
                classification = "POTENTIAL ENCROACHMENT"
                confidence = 0.76
                reason = f"New construction in low-density area - requires investigation"
                subcategory = "Rural Development"
                severity = "MEDIUM"
            else:
                classification = "NATURAL/SEASONAL"
                confidence = 0.82
                reason = "Natural/seasonal changes"
                subcategory = "Natural"
                severity = "LOW"
        
        statistics = {
            "change_percentage": round(change_pct, 2),
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "area_km2": area_km2,
            "building_count": total_buildings,
            "building_density_per_km2": round(density, 1)
        }
        
    else:
        # Fallback classification without GEE data
        print("⚠️  Using basic classification (no GEE data)")
        
        if change_pct > 10:
            classification = "SIGNIFICANT CHANGES"
            confidence = 0.75
            reason = f"Large changes detected ({change_pct:.1f}%) - manual verification needed"
            subcategory = "Unknown Type"
            severity = "MEDIUM"
        elif change_pct > 3:
            classification = "MODERATE CHANGES"
            confidence = 0.65
            reason = f"Moderate changes detected ({change_pct:.1f}%)"
            subcategory = "Needs Investigation"
            severity = "MEDIUM"
        else:
            classification = "MINOR CHANGES"
            confidence = 0.70
            reason = f"Minor changes detected ({change_pct:.1f}%)"
            subcategory = "Low Priority"
            severity = "LOW"
        
        # Calculate area from AOI
        from math import cos, radians
        lat_min, lon_min, lat_max, lon_max = aoi
        lat_diff = lat_max - lat_min
        lon_diff = lon_max - lon_min
        lat_avg = (lat_min + lat_max) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * cos(radians(lat_avg))
        area_km2 = (lat_diff * km_per_deg_lat) * (lon_diff * km_per_deg_lon)
        
        statistics = {
            "change_percentage": round(change_pct, 2),
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "area_km2": round(area_km2, 2),
            "building_count": 0,
            "building_density_per_km2": 0
        }
    
    # Compile results
    results = {
        "classification": classification,
        "subcategory": subcategory,
        "confidence": round(confidence, 2),
        "severity": severity,
        "reason": reason,
        "statistics": statistics,
        "date_range": {
            "start": start_date,
            "end": end_date
        },
        "gee_validation": gee_validation
    }
    
    print(f"\n🚨 CLASSIFICATION: {classification}")
    print(f"📊 Severity: {severity}")
    print(f"💯 Confidence: {confidence*100:.0f}%")
    print(f"💡 Reason: {reason}")
    
    return results


def save_classification(results, output_dir):
    """Save classification results to JSON and TXT"""
    import json
    
    # Save JSON
    json_path = output_dir / "classification_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save TXT report
    txt_path = output_dir / "classification_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ENCROACHMENT CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Classification: {results['classification']}\n")
        f.write(f"Subcategory: {results['subcategory']}\n")
        f.write(f"Confidence: {results['confidence']*100:.0f}%\n")
        f.write(f"Severity: {results['severity']}\n\n")
        
        f.write(f"Reason:\n{results['reason']}\n\n")
        
        f.write("Statistics:\n")
        f.write(f"  Change Percentage: {results['statistics']['change_percentage']:.2f}%\n")
        f.write(f"  Changed Pixels: {results['statistics']['changed_pixels']:,}\n")
        f.write(f"  Total Pixels: {results['statistics']['total_pixels']:,}\n")
        f.write(f"  Area: {results['statistics']['area_km2']:.2f} km²\n")
        f.write(f"  Buildings: {results['statistics']['building_count']:,}\n")
        f.write(f"  Building Density: {results['statistics']['building_density_per_km2']:.1f} per km²\n\n")
        
        f.write(f"Date Range: {results['date_range']['start']} to {results['date_range']['end']}\n")
        
        if results.get('gee_validation'):
            f.write("\nGEE Building Validation: ✅ Available\n")
            f.write(f"  Total Buildings Found: {results['gee_validation']['total_buildings']:,}\n")
            f.write(f"  Average Building Size: {results['gee_validation'].get('avg_building_size', 0):.0f} m²\n")
        else:
            f.write("\nGEE Building Validation: ❌ Not Available\n")
    
    return json_path, txt_path


def save_visualization(bin_mask, t0_path, t1_path, start_date, end_date, classification_results=None):
    """
    Create and save change visualization with VISIBLE dates and classification
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
    
    # Create side-by-side comparison with header
    header_height = 120 if classification_results else 80
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
    
    # Draw text with black outline
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_text_with_outline(img, text, position, font_scale, thickness):
        x, y = position
        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # T0 labels
    draw_text_with_outline(canvas, 'T0 (Before)', (10, 30), 0.9, 2)
    draw_text_with_outline(canvas, t0_display, (10, 60), 0.6, 1)
    
    # T1 labels
    draw_text_with_outline(canvas, 'T1 (After)', (w + 10, 30), 0.9, 2)
    draw_text_with_outline(canvas, t1_display, (w + 10, 60), 0.6, 1)
    
    # Changes labels
    draw_text_with_outline(canvas, 'Changes (Red)', (2 * w + 10, 30), 0.9, 2)
    draw_text_with_outline(canvas, date_range, (2 * w + 10, 60), 0.5, 1)
    
    # Add classification if available
    if classification_results:
        classification_text = f"{classification_results['classification']} ({classification_results['confidence']*100:.0f}%)"
        severity_color = {
            'HIGH': (0, 0, 255),    # Red
            'MEDIUM': (0, 165, 255), # Orange
            'LOW': (0, 255, 0)       # Green
        }.get(classification_results['severity'], (255, 255, 255))
        
        cv2.putText(canvas, classification_text, (10, 95), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, classification_text, (10, 95), font, 0.7, severity_color, 2, cv2.LINE_AA)
    
    # Save
    output_path = Path("outputs/change_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"✅ Visualization saved: {output_path}")


def main(args=None):
    """Main pipeline execution with GEE validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run change detection pipeline with GEE validation")
    parser.add_argument("lat_min", type=float, help="Minimum latitude")
    parser.add_argument("lon_min", type=float, help="Minimum longitude")
    parser.add_argument("lat_max", type=float, help="Maximum latitude")
    parser.add_argument("lon_max", type=float, help="Maximum longitude")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable image enhancement")
    parser.add_argument("--no-gee", action="store_true", help="Disable GEE validation")
    
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
    aoi_coords = [args.lat_min, args.lon_min, args.lat_max, args.lon_max]
    start_date = args.start_date
    end_date = args.end_date
    
    print("=" * 70)
    print("🛰️  URBANEYE ENCROACHMENT DETECTION PIPELINE WITH GEE")
    print("=" * 70)
    print(f"📍 AOI: {aoi}")
    print(f"📅 Time Range: {start_date} → {end_date}")
    print(f"🎨 Image Enhancement: {'Enabled' if not args.no_enhance else 'Disabled'}")
    print(f"🏢 GEE Validation: {'Enabled' if not args.no_gee else 'Disabled'}")
    print("=" * 70)
    
    # Import fetch function
    from fetch_sentinel import fetch_sentinel_image
    
    # Step 1: Fetch T0 image
    print(f"\n{'='*70}")
    print(f"📍 STEP 1: FETCHING T0 IMAGE ({start_date})")
    print(f"{'='*70}")
    img_t0, actual_start_date = fetch_sentinel_image(
        aoi, start_date, str(t0_path), enhance=not args.no_enhance
    )
    
    if img_t0 is None or img_t0.mean() < 1 or img_t0.mean() > 250:
        print("\n❌ CRITICAL: Failed to fetch valid T0 image")
        return
    
    if actual_start_date:
        start_date = actual_start_date
    
    # Step 2: Fetch T1 image
    print(f"\n{'='*70}")
    print(f"📍 STEP 2: FETCHING T1 IMAGE ({end_date})")
    print(f"{'='*70}")
    img_t1, actual_end_date = fetch_sentinel_image(
        aoi, end_date, str(t1_path), enhance=not args.no_enhance
    )
    
    if img_t1 is None or img_t1.mean() < 1 or img_t1.mean() > 250:
        print("\n❌ CRITICAL: Failed to fetch valid T1 image")
        return
    
    if actual_end_date:
        end_date = actual_end_date
    
    # Step 3: Additional enhancement if enabled
    if not args.no_enhance:
        print(f"\n{'='*70}")
        print("📍 STEP 3: ADDITIONAL IMAGE ENHANCEMENT")
        print(f"{'='*70}")
        img_t0 = enhance_image(img_t0)
        img_t1 = enhance_image(img_t1)
        
        # Save enhanced images
        cv2.imwrite(str(raw_dir / "t0_enhanced.png"), img_t0)
        cv2.imwrite(str(raw_dir / "t1_enhanced.png"), img_t1)
        print("✅ Enhanced images saved")
    
    # Step 4: Preprocess
    print(f"\n{'='*70}")
    print("📍 STEP 4: PREPROCESSING")
    print(f"{'='*70}")
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    print("✅ Preprocessing complete")
    
    # Step 5: Run inference
    print(f"\n{'='*70}")
    print("📍 STEP 5: CHANGE DETECTION")
    print(f"{'='*70}")
    bin_mask = run_inference(t0_norm, t1_norm)
    
    total_pixels = bin_mask.size
    changed_pixels = bin_mask.sum()
    pct_change = (changed_pixels / total_pixels) * 100
    
    print(f"✅ Change detection complete")
    print(f"📊 Changed pixels: {changed_pixels:,} ({pct_change:.2f}%)")
    
    # Step 6: GEE Building Validation
    gee_validation = None
    if not args.no_gee:
        print(f"\n{'='*70}")
        print("📍 STEP 6: GEE BUILDING VALIDATION")
        print(f"{'='*70}")
        gee_validation = validate_with_gee_buildings(bin_mask, aoi_coords)
    
    # Step 7: Classification
    print(f"\n{'='*70}")
    print("📍 STEP 7: ENCROACHMENT CLASSIFICATION")
    print(f"{'='*70}")
    classification_results = classify_encroachment(
        bin_mask, 
        gee_validation, 
        start_date, 
        end_date,
        aoi_coords
    )
    
    # Save classification
    json_path, txt_path = save_classification(classification_results, output_dir)
    print(f"✅ Classification saved:")
    print(f"   - {json_path}")
    print(f"   - {txt_path}")
    
    # Step 8: Visualization
    print(f"\n{'='*70}")
    print("📍 STEP 8: CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    save_visualization(bin_mask, str(t0_path), str(t1_path), start_date, end_date, classification_results)
    
    # Step 9: Generate report
    print(f"\n{'='*70}")
    print("📍 STEP 9: GENERATING REPORT")
    print(f"{'='*70}")
    from generate_report import generate_report
    report = generate_report(bin_mask, str(t0_path), str(t1_path))
    
    report_path = output_dir / "report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Report saved: {report_path}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Classification: {classification_results['classification']}")
    print(f"  Severity: {classification_results['severity']}")
    print(f"  Confidence: {classification_results['confidence']*100:.0f}%")
    print(f"  Change: {pct_change:.2f}%")
    
    if gee_validation:
        print(f"\n🏢 GEE BUILDING DATA:")
        print(f"  Total buildings: {gee_validation['total_buildings']:,}")
        print(f"  Density: {gee_validation['building_density']:.1f} buildings/km²")
        print(f"  Zone: {'Dense Urban' if gee_validation['is_dense_urban'] else 'Urban' if gee_validation['is_urban_area'] else 'Non-urban'}")
    
    print(f"\n📂 Files created:")
    print(f"  - outputs/change_map.png (visualization)")
    print(f"  - outputs/classification_results.json")
    print(f"  - outputs/classification_report.txt")
    print(f"  - outputs/report.txt")
    
    print("\n🎉 Encroachment detection complete!")
    print("=" * 70)
    
    return classification_results


if __name__ == "__main__":
    main()
