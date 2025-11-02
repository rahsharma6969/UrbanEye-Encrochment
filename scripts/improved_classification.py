# scripts/improved_classification.py
"""
IMPROVED Classification using multiple data sources
"""

import numpy as np
import requests
from math import cos, radians


def get_building_density_estimate(aoi):
    """
    Improved building density estimation
    Uses multiple heuristics when API data is incomplete
    """
    
    lon_min, lat_min, lon_max, lat_max = aoi
    
    # Calculate area
    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    lat_avg = (lat_min + lat_max) / 2
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * cos(radians(lat_avg))
    area_km2 = (lat_diff * km_per_deg_lat) * (lon_diff * km_per_deg_lon)
    
    # Try Overpass API with longer timeout
    building_count = query_osm_buildings(aoi)
    
    if building_count < 5:  # Likely incomplete data
        print("⚠️  OSM data incomplete, using density estimation...")
        building_count = estimate_from_coordinates(lat_avg, lon_avg, area_km2)
    
    density = building_count / area_km2 if area_km2 > 0 else 0
    
    return building_count, density, area_km2


def query_osm_buildings(aoi, timeout=60):
    """Query OSM with longer timeout"""
    
    lon_min, lat_min, lon_max, lat_max = aoi
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out count;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=timeout+10)
        if response.status_code == 200:
            data = response.json()
            count = len(data.get('elements', []))
            print(f"✅ OSM returned {count} buildings")
            return count
    except:
        pass
    
    return 0


def estimate_from_coordinates(lat, lon, area_km2):
    """
    Estimate building density based on known urban densities in India
    """
    
    # Mumbai region coordinates (approximate)
    mumbai_center = (19.076, 72.877)
    
    # Calculate distance from Mumbai center
    dist = ((lat - mumbai_center[0])**2 + (lon - mumbai_center[1])**2)**0.5
    
    # Urban density zones for Mumbai region
    if dist < 0.05:  # < 5km from center (Dense urban)
        density = 350  # buildings/km²
        zone = "Dense Urban (CBD)"
    elif dist < 0.10:  # 5-10km (Urban)
        density = 200
        zone = "Urban"
    elif dist < 0.20:  # 10-20km (Suburban)
        density = 80
        zone = "Suburban"
    elif dist < 0.40:  # 20-40km (Peri-urban)
        density = 30
        zone = "Peri-urban"
    else:  # > 40km (Rural)
        density = 5
        zone = "Rural"
    
    estimated_count = int(area_km2 * density)
    
    print(f"📍 Estimated zone: {zone}")
    print(f"📊 Estimated density: {density} buildings/km²")
    print(f"🏗️  Estimated buildings: {estimated_count}")
    
    return estimated_count


def classify_with_improved_logic(change_mask, aoi, start_date, end_date):
    """
    IMPROVED classification with better building detection
    """
    
    print("\n" + "="*70)
    print("🔍 IMPROVED CLASSIFICATION")
    print("="*70)
    
    # Get change stats
    total_pixels = change_mask.size
    changed_pixels = int(np.sum(change_mask))
    change_pct = (changed_pixels / total_pixels) * 100
    
    print(f"📊 Changes: {changed_pixels:,} pixels ({change_pct:.2f}%)")
    
    # Get building data
    building_count, density, area_km2 = get_building_density_estimate(aoi)
    
    print(f"🏘️  Building density: {density:.1f} per km²")
    print(f"📐 Area: {area_km2:.2f} km²")
    
    # IMPROVED CLASSIFICATION LOGIC
    # Based on actual urban planning standards
    
    if density > 200:  # Dense urban
        if change_pct > 2:
            classification = "MAN-MADE"
            confidence = 0.92
            reason = f"Dense urban area ({density:.0f} buildings/km²) with {change_pct:.1f}% change - likely construction/demolition"
            subcategory = "Urban Development"
        else:
            classification = "MAN-MADE"
            confidence = 0.78
            reason = f"Dense urban area with minor infrastructure changes"
            subcategory = "Urban Maintenance"
    
    elif density > 80:  # Urban/Suburban
        if change_pct > 5:
            classification = "MAN-MADE"
            confidence = 0.88
            reason = f"Urban area ({density:.0f} buildings/km²) with significant expansion ({change_pct:.1f}%)"
            subcategory = "Urban Expansion"
        elif change_pct > 2:
            classification = "MIXED"
            confidence = 0.72
            reason = f"Moderate urban area with mixed changes"
            subcategory = "Mixed Development"
        else:
            classification = "NATURAL"
            confidence = 0.68
            reason = f"Urban area with minor seasonal changes"
            subcategory = "Seasonal/Natural"
    
    elif density > 30:  # Peri-urban
        if change_pct > 8:
            classification = "MAN-MADE"
            confidence = 0.82
            reason = f"Peri-urban development detected ({change_pct:.1f}% change)"
            subcategory = "Peri-urban Development"
        elif change_pct > 3:
            classification = "MIXED"
            confidence = 0.70
            reason = f"Mixed urban-natural changes in developing area"
            subcategory = "Mixed Development"
        else:
            classification = "NATURAL"
            confidence = 0.75
            reason = f"Primarily natural/agricultural changes"
            subcategory = "Agricultural/Natural"
    
    else:  # Rural
        if change_pct > 15:
            classification = "NATURAL"
            confidence = 0.86
            reason = f"Large natural changes in rural area ({change_pct:.1f}%)"
            subcategory = "Natural/Agricultural"
        elif change_pct > 5 and density > 10:
            classification = "MAN-MADE"
            confidence = 0.76
            reason = f"Rural development/settlement expansion"
            subcategory = "Rural Development"
        else:
            classification = "NATURAL"
            confidence = 0.82
            reason = f"Natural/seasonal changes in rural area"
            subcategory = "Seasonal/Natural"
    
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
            "building_density_per_km2": round(density, 1)
        },
        "date_range": {
            "start": start_date,
            "end": end_date
        }
    }
    
    print("\n" + "="*70)
    print(f"✅ CLASSIFICATION: {classification}")
    print("="*70)
    print(f"💯 Confidence: {confidence*100:.0f}%")
    print(f"💡 Reason: {reason}")
    
    return results
