# scripts/enhanced_visualization.py (FIXED VERSION)
"""
Enhanced visualization with color-coded Natural vs Man-Made classification
🟢 Green = Natural changes
🔴 Red = Man-made changes
"""

import cv2
import numpy as np
from pathlib import Path
import json


def classify_change_regions(bin_mask, aoi, start_date, end_date):
    """
    Classify individual change regions as natural or man-made
    """
    from classify_with_overture import classify_changes_no_auth
    
    # Get overall classification
    overall_result = classify_changes_no_auth(bin_mask, aoi, start_date, end_date)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8
    )
    
    regions = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < 50:
            continue
        
        region_mask = (labels == i).astype(np.uint8)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
        
        # Classification heuristic
        aspect_ratio = w / h if h > 0 else 1
        compactness = (4 * np.pi * area) / ((w + h) ** 2) if (w + h) > 0 else 0
        
        if overall_result['classification'] == 'MAN-MADE':
            if 0.5 < aspect_ratio < 2.0 and compactness > 0.3:
                classification = 'MAN-MADE'
            else:
                classification = 'NATURAL'
        elif overall_result['classification'] == 'NATURAL':
            if compactness < 0.2 or aspect_ratio > 3:
                classification = 'NATURAL'
            else:
                classification = 'MAN-MADE'
        else:
            if 0.7 < aspect_ratio < 1.5 and compactness > 0.4:
                classification = 'MAN-MADE'
            else:
                classification = 'NATURAL'
        
        regions.append({
            'mask': region_mask,
            'classification': classification,
            'bbox': (x, y, w, h),
            'area': area,
            'centroid': centroids[i]
        })
    
    return regions, overall_result


def create_enhanced_visualization(bin_mask, t0_path, t1_path, start_date, end_date, aoi):
    """
    Create color-coded visualization with labeled changes
    """
    
    print("\n🎨 Creating enhanced visualization...")
    
    # Load images
    img1 = cv2.imread(str(t0_path))
    img2 = cv2.imread(str(t1_path))
    
    if img1 is None or img2 is None:
        print("❌ Failed to load images")
        return None, None
    img1 = cv2.resize(img1, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    h, w = h*2, w*2
    bin_mask = cv2.resize(bin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Ensure same size
    h = min(img1.shape[0], img2.shape[0], bin_mask.shape[0])
    w = min(img1.shape[1], img2.shape[1], bin_mask.shape[1])
    
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    bin_mask = cv2.resize(bin_mask.astype(np.uint8), (w, h))
    
    # Classify regions
    print("🔍 Classifying change regions...")
    regions, overall_result = classify_change_regions(bin_mask, aoi, start_date, end_date)
    
    print(f"✅ Found {len(regions)} change regions")
    
    # Create result overlay
    result = img2.copy()
    
    # Separate masks
    manmade_mask = np.zeros((h, w), dtype=np.uint8)
    natural_mask = np.zeros((h, w), dtype=np.uint8)
    
    for region in regions:
        if region['classification'] == 'MAN-MADE':
            manmade_mask = cv2.bitwise_or(manmade_mask, region['mask'])
        else:
            natural_mask = cv2.bitwise_or(natural_mask, region['mask'])
    
    # ✨ FIXED: Apply colored overlays correctly
    # Red for man-made
    red_overlay = np.zeros_like(result)
    red_overlay[:, :, 2] = manmade_mask * 255  # Red channel
    result = cv2.addWeighted(result, 0.7, red_overlay, 0.3, 0)
    
    # Green for natural
    green_overlay = np.zeros_like(result)
    green_overlay[:, :, 1] = natural_mask * 255  # Green channel
    result = cv2.addWeighted(result, 0.7, green_overlay, 0.3, 0)
    
    # Add text labels for major changes
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled_result = result.copy()
    
    for region in sorted(regions, key=lambda r: r['area'], reverse=True)[:5]:
        cx, cy = int(region['centroid'][0]), int(region['centroid'][1])
        label = region['classification']
        color = (0, 0, 255) if label == 'MAN-MADE' else (0, 255, 0)
        
        # Ensure position is within bounds
        if 10 < cx < w-100 and 20 < cy < h-20:
            # Background box
            text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
            cv2.rectangle(labeled_result,
                         (cx - 5, cy - text_size[1] - 5),
                         (cx + text_size[0] + 5, cy + 5),
                         (0, 0, 0), -1)
            # Text
            cv2.putText(labeled_result, label, (cx, cy), font, 0.5, color, 1, cv2.LINE_AA)
    
    # Create 4-panel canvas
    header_h = 100
    canvas = np.zeros((h + header_h, w*4, 3), dtype=np.uint8)
    
    # Gradient header
    for i in range(header_h):
        intensity = int(50 * (1 - i/header_h))
        canvas[i, :] = [intensity, intensity, intensity]
    
    # Place images
    canvas[header_h:, :w] = img1
    canvas[header_h:, w:2*w] = img2
    canvas[header_h:, 2*w:3*w] = labeled_result
    
    # Create legend panel
    legend = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(legend, "LEGEND", (20, 30), font, 0.8, (0, 0, 0), 2)
    
    # Legend items
    y = 70
    cv2.rectangle(legend, (20, y-15), (50, y+5), (0, 0, 255), -1)
    cv2.putText(legend, "Man-made", (60, y), font, 0.6, (0, 0, 0), 1)
    
    y += 35
    cv2.rectangle(legend, (20, y-15), (50, y+5), (0, 255, 0), -1)
    cv2.putText(legend, "Natural", (60, y), font, 0.6, (0, 0, 0), 1)
    
    # Statistics
    y += 60
    cv2.putText(legend, "STATISTICS", (20, y), font, 0.8, (0, 0, 0), 2)
    y += 35
    
    stats_text = [
        f"Total: {bin_mask.sum():,} px",
        f"Change: {(bin_mask.sum()/bin_mask.size*100):.1f}%",
        "",
        f"Man-made: {manmade_mask.sum():,} px",
        f"Natural: {natural_mask.sum():,} px",
        "",
        "CLASSIFICATION",
        f"{overall_result['classification']}",
        f"{overall_result['confidence']*100:.0f}% confidence",
        "",
        f"Buildings: {overall_result['statistics']['building_count']}",
        f"Density: {overall_result['statistics']['building_density_per_km2']:.1f}/km²",
        f"Area: {overall_result['statistics']['area_km2']:.2f} km²"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(legend, text, (20, y + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    canvas[header_h:, 3*w:] = legend
    
    # Headers
    def draw_header(img, text, x, y, scale=1.0):
        cv2.putText(img, text, (x+2, y+2), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    
    try:
        from datetime import datetime
        t0_obj = datetime.strptime(start_date, "%Y-%m-%d")
        t1_obj = datetime.strptime(end_date, "%Y-%m-%d")
        t0_display = t0_obj.strftime("%b %d, %Y")
        t1_display = t1_obj.strftime("%b %d, %Y")
    except:
        t0_display = start_date
        t1_display = end_date
    
    draw_header(canvas, 'T0 (Before)', 15, 35, 0.8)
    draw_header(canvas, t0_display, 15, 70, 0.6)
    
    draw_header(canvas, 'T1 (After)', w+15, 35, 0.8)
    draw_header(canvas, t1_display, w+15, 70, 0.6)
    
    draw_header(canvas, 'Classified Changes', 2*w+15, 35, 0.8)
    draw_header(canvas, 'Red=Man Green=Natural', 2*w+15, 70, 0.5)
    
    draw_header(canvas, 'Statistics', 3*w+15, 35, 0.8)
    
    # Save
    output_path = Path("outputs/enhanced_change_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    
    print(f"\n✅ Enhanced visualization saved: {output_path}")
    print(f"   🔴 Man-made: {manmade_mask.sum():,} pixels")
    print(f"   🟢 Natural: {natural_mask.sum():,} pixels")
    
    return canvas, overall_result


if __name__ == "__main__":
    print("🧪 Testing enhanced visualization...")
    
    raw_dir = Path("data/raw")
    t0_path = raw_dir / "t0.png"
    t1_path = raw_dir / "t1.png"
    
    if not t0_path.exists() or not t1_path.exists():
        print("❌ Run pipeline first")
        exit(1)
    
    # Run change detection
    from run_pipeline import preprocess_image, run_inference
    
    img_t0 = cv2.imread(str(t0_path))
    img_t1 = cv2.imread(str(t1_path))
    
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    
    bin_mask = run_inference(t0_norm, t1_norm)
    
    # Create visualization
    aoi = [72.86, 19.05, 72.88, 19.08]
    start_date = "2024-01-15"
    end_date = "2024-07-15"
    
    canvas, results = create_enhanced_visualization(
        bin_mask, str(t0_path), str(t1_path),
        start_date, end_date, aoi
    )
    
    if canvas is not None:
        print("\n✅ Test complete!")
        print("📂 Check: outputs/enhanced_change_map.png")
    else:
        print("\n❌ Failed to create visualization")
