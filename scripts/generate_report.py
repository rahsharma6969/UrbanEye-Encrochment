# ml/scripts/generate_report.py
"""
Generate natural language explanation of detected changes
"""

import numpy as np
from pathlib import Path


def generate_report(bin_mask, t0_path, t1_path):
    """Generate text report"""
    total_pixels = bin_mask.size
    changed_pixels = bin_mask.sum()
    pct_change = (changed_pixels / total_pixels) * 100

    # Find regions
    try:
        import cv2
        contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_regions = len(contours)
        avg_size = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
    except:
        num_regions = int(changed_pixels // 1000)
        avg_size = 0

    # Interpretation
    if pct_change < 0.5:
        status = "✅ MINIMAL CHANGE"
        interpretation = "No significant development detected."
    elif pct_change < 5.0:
        status = "⚠️ MINOR CHANGE"
        interpretation = "Small-scale construction or land use changes detected."
    elif pct_change < 15.0:
        status = "🟡 MODERATE CHANGE"
        interpretation = "New buildings, roads, or parking lots likely added."
    else:
        status = "🔴 SIGNIFICANT URBAN EXPANSION"
        interpretation = "Major infrastructure growth — likely new housing or industrial zones."

    report = [
        "📊 CHANGE DETECTION REPORT",
        "-"*60,
        f"📸 Dates:",
        f"   Before: {Path(t0_path).name}",
        f"   After:  {Path(t1_path).name}",
        "",
        "📈 STATISTICS",
        f"   Total pixels:     {total_pixels:,}",
        f"   Changed pixels:   {changed_pixels:,}",
        f"   Change %:         {pct_change:.2f}%",
        f"   Detected regions: {num_regions}",
        "",
        f"🎯 ASSESSMENT: {status}",
        f"   {interpretation}",
        "",
        f"💾 Visualization saved to: outputs/change_map.png"
    ]

    return "\n".join(report)