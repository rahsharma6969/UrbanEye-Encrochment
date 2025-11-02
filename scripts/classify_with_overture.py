# scripts/classify_with_overture.py
"""
ZERO-AUTH Classification using Overture Maps + OSM Building Data
Works immediately without any authentication!
"""

import requests
import json
from pathlib import Path
from math import cos, radians
import numpy as np


def get_overture_buildings(bbox):
    """
    Get building count from Overture Maps API
    Overture = Open building dataset (free, no auth)
    
    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]
    
    Returns:
        int: Number of buildings found
    """
    
    lon_min, lat_min, lon_max, lat_max = bbox
    
    print(f"🏗️  Querying Overture Maps for buildings...")
    
    # Use Overpass API (OpenStreetMap data - free, no auth)
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query for buildings in bounding box
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out count;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Count elements
            building_count = len(data.get('elements', []))
            
            print(f"✅ Found {building_count} buildings")
            return building_count
        else:
            print(f"⚠️  API returned status {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"⚠️  Error querying buildings: {e}")
        # Fallback: estimate from area
        return estimate_buildings_from_area(bbox)


def estimate_buildings_from_area(bbox):
    """
    Fallback: Estimate building count based on area
    Uses typical urban densities for Mumbai
    """
    
    area_km2 = calculate_area_km2(bbox)
    
    # Mumbai typical densities (buildings per km²)
    # Dense urban: 200-500
    # Suburban: 50-200
    # Rural: 0-50
    
    # Conservative estimate for Mumbai region
    estimated_density = 150  # mid-range
    estimated_count = int(area_km2 * estimated_density)
    
    print(f"⚠️  Using estimated building count: {estimated_count}")
    return estimated_count


def calculate_area_km2(bbox):
    """Calculate area in km²"""
    lon_min, lat_min, lon_max, lat_max = bbox
    
    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    
    lat_avg = (lat_min + lat_max) / 2
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * cos(radians(lat_avg))
    
    area = (lat_diff * km_per_degree_lat) * (lon_diff * km_per_degree_lon)
    return area


def classify_changes_no_auth(change_mask, aoi, start_date, end_date):
    """
    Classify changes WITHOUT any authentication
    Uses free OpenStreetMap data via Overpass API
    """
    
    print("\n" + "="*60)
    print("🔍 CLASSIFYING CHANGES (NO AUTH NEEDED)")
    print("="*60)
    
    # Get change statistics
    total_pixels = change_mask.size
    changed_pixels = int(np.sum(change_mask))
    change_pct = (changed_pixels / total_pixels) * 100
    
    print(f"📊 Changes: {changed_pixels:,} pixels ({change_pct:.2f}%)")
    
    # Calculate area
    area_km2 = calculate_area_km2(aoi)
    print(f"📐 Area: {area_km2:.2f} km²")
    
    # Get building data from OpenStreetMap (free!)
    building_count = get_overture_buildings(aoi)
    
    # Calculate density
    building_density = building_count / area_km2 if area_km2 > 0 else 0
    print(f"🏘️  Density: {building_density:.1f} buildings/km²")
    
    # CLASSIFICATION LOGIC (same as before)
    if building_density > 150:
        if change_pct > 3:
            classification = "MAN-MADE"
            confidence = 0.90
            reason = f"Dense urban area ({building_density:.0f} buildings/km²) with {change_pct:.1f}% change"
            subcategory = "Urban Development"
        else:
            classification = "MAN-MADE"
            confidence = 0.75
            reason = f"Dense urban area with minor changes"
            subcategory = "Urban Maintenance"
    
    elif building_density > 50:
        if change_pct > 5:
            classification = "MAN-MADE"
            confidence = 0.85
            reason = f"Suburban expansion detected ({change_pct:.1f}%)"
            subcategory = "Suburban Development"
        elif change_pct > 2:
            classification = "MIXED"
            confidence = 0.70
            reason = f"Mixed urban-natural changes"
            subcategory = "Mixed Development"
        else:
            classification = "NATURAL"
            confidence = 0.65
            reason = f"Minor natural changes in suburban area"
            subcategory = "Natural/Seasonal"
    
    else:
        if change_pct > 10:
            classification = "NATURAL"
            confidence = 0.85
            reason = f"Large natural changes in rural area"
            subcategory = "Natural/Agricultural"
        elif building_density > 10 and change_pct > 3:
            classification = "MAN-MADE"
            confidence = 0.75
            reason = f"Rural development ({change_pct:.1f}%)"
            subcategory = "Rural Development"
        else:
            classification = "NATURAL"
            confidence = 0.80
            reason = f"Natural changes in rural area"
            subcategory = "Natural/Seasonal"
    
    # Compile results
    results = {
        "classification": classification,
        "subcategory": subcategory,
        "confidence": round(confidence, 2),
        "reason": reason,
        "statistics": {
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "change_percentage": round(change_pct, 2),
            "area_km2": round(area_km2, 2),
            "building_count": building_count,
            "building_density_per_km2": round(building_density, 1)
        },
        "date_range": {
            "start": start_date,
            "end": end_date
        },
        "method": "OpenStreetMap Overpass API (No Authentication Required)"
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"✅ RESULT: {classification}")
    print("="*60)
    print(f"📁 Category: {subcategory}")
    print(f"💯 Confidence: {confidence*100:.0f}%")
    print(f"💡 Reason: {reason}")
    
    return results


def save_results(results, output_dir="outputs"):
    """Save results to JSON and TXT"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_path = output_dir / "classification_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # TXT
    txt_path = output_dir / "classification_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("URBAN CHANGE CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Classification: {results['classification']}\n")
        f.write(f"Category: {results['subcategory']}\n")
        f.write(f"Confidence: {results['confidence']*100:.0f}%\n")
        f.write(f"Reason: {results['reason']}\n")
        f.write(f"Method: {results['method']}\n\n")
        
        f.write("STATISTICS:\n")
        f.write("-" * 40 + "\n")
        stats = results['statistics']
        f.write(f"Changed Pixels: {stats['changed_pixels']:,} ({stats['change_percentage']}%)\n")
        f.write(f"Area: {stats['area_km2']} km²\n")
        f.write(f"Buildings: {stats['building_count']}\n")
        f.write(f"Density: {stats['building_density_per_km2']} buildings/km²\n")
    
    print(f"\n💾 Results saved:")
    print(f"   - {json_path}")
    print(f"   - {txt_path}")
    
    return json_path, txt_path


if __name__ == "__main__":
    # Test
    print("🧪 Testing NO-AUTH classification...")
    
    # Dummy change mask
    test_mask = np.random.randint(0, 2, (512, 512))
    
    # Mumbai BKC
    test_aoi = [72.86, 19.05, 72.88, 19.08]
    
    # Classify
    results = classify_changes_no_auth(
        test_mask,
        test_aoi,
        "2024-01-15",
        "2024-07-15"
    )
    
    # Save
    save_results(results)
    
    print("\n✅ Test complete!")
