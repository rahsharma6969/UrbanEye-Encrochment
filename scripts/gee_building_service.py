"""
Google Earth Engine Building Detection Service
Replaces OSM Overpass API with 100x better data
"""

import ee
import json
from pathlib import Path
from typing import Dict
import time

class GEEBuildingService:
    """Service to fetch building footprints from Google Earth Engine"""
    
    def __init__(self, project_id: str = 'gee project id'):  # ← Your working project ID
        """
        Initialize Earth Engine
        
        Args:
            project_id: Your GEE project ID
        """
        try:
            ee.Initialize(project=project_id)
            print(f"✅ Google Earth Engine initialized (Project: {project_id})")
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            raise
        
        # Load Open Buildings dataset
        self.open_buildings = ee.FeatureCollection(
            'GOOGLE/Research/open-buildings/v3/polygons'
        )
        print("✅ Open Buildings dataset loaded")
    
    def get_buildings_for_area(
        self, 
        lat_min: float, 
        lon_min: float, 
        lat_max: float, 
        lon_max: float,
        confidence_threshold: float = 0.75
    ) -> Dict:
        """
        Fetch building footprints for given area
        
        Args:
            lat_min, lon_min, lat_max, lon_max: Bounding box coordinates
            confidence_threshold: Minimum building confidence (0.65-1.0)
            
        Returns:
            Dictionary with building statistics and features
        """
        
        print(f"\n{'='*70}")
        print(f"🔍 FETCHING BUILDINGS FROM GOOGLE EARTH ENGINE")
        print(f"{'='*70}")
        print(f"📍 Area: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        
        # Create bounding box
        aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        
        # Calculate area
        area_m2 = aoi.area().getInfo()
        area_km2 = area_m2 / 1_000_000
        print(f"📐 Area: {area_km2:.2f} km²")
        
        # Filter buildings
        buildings_in_area = self.open_buildings.filterBounds(aoi)
        
        # Apply confidence filter
        buildings_filtered = buildings_in_area.filter(
            ee.Filter.gte('confidence', confidence_threshold)
        )
        
        # Get count
        total_count = buildings_filtered.size().getInfo()
        print(f"🏢 Buildings found: {total_count:,}")
        
        if total_count == 0:
            print("⚠️  No buildings found. This area may not be covered by Open Buildings.")
            print("💡 Coverage: Africa, Latin America, South/Southeast Asia")
            return {
                'total_buildings': 0,
                'area_km2': area_km2,
                'density_per_km2': 0,
                'features': []
            }
        
        # Calculate density
        density = total_count / area_km2 if area_km2 > 0 else 0
        print(f"📊 Density: {density:.1f} buildings/km²")
        
        # Get building statistics
        stats = self._get_building_stats(buildings_filtered, aoi)
        
        # Export limited set of features for analysis (max 5000 for performance)
        limit = min(total_count, 5000)
        print(f"⏳ Exporting {limit} building features (this may take 10-20 seconds)...")
        features = buildings_filtered.limit(limit).getInfo()
        
        print(f"✅ Exported {len(features['features'])} building features")
        
        result = {
            'total_buildings': total_count,
            'area_km2': area_km2,
            'density_per_km2': round(density, 1),
            'bounds': [lon_min, lat_min, lon_max, lat_max],
            'confidence_threshold': confidence_threshold,
            'statistics': stats,
            'features': features['features']
        }
        
        print(f"{'='*70}\n")
        return result
    
    def _get_building_stats(self, buildings_fc, aoi):
        """Calculate building statistics"""
        
        print("📊 Calculating statistics...")
        
        # Total building area
        total_area = buildings_fc.aggregate_sum('area_in_meters').getInfo()
        
        # Average building size
        avg_area = buildings_fc.aggregate_mean('area_in_meters').getInfo()
        
        # Confidence distribution
        conf_stats = {
            'mean': buildings_fc.aggregate_mean('confidence').getInfo(),
            'min': buildings_fc.aggregate_min('confidence').getInfo(),
            'max': buildings_fc.aggregate_max('confidence').getInfo()
        }
        
        return {
            'total_building_area_m2': round(total_area, 2) if total_area else 0,
            'average_building_size_m2': round(avg_area, 2) if avg_area else 0,
            'confidence': conf_stats
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save building data to JSON file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        return output_file
    
    def export_to_geojson(self, results: Dict, output_path: str):
        """Export buildings to GeoJSON format"""
        
        geojson = {
            'type': 'FeatureCollection',
            'metadata': {
                'total_buildings': results['total_buildings'],
                'area_km2': results['area_km2'],
                'density_per_km2': results['density_per_km2'],
                'bounds': results['bounds'],
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'features': results['features']
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"💾 GeoJSON exported to: {output_file}")
        return output_file


def main():
    """Example usage - Mumbai Anand Nagar"""
    
    print("\n" + "="*70)
    print("🚀 TESTING GEE BUILDING SERVICE")
    print("="*70 + "\n")
    
    # Initialize service
    service = GEEBuildingService(project_id='urbaneye-476904')
    
    # Mumbai Anand Nagar coordinates (from your screenshot)
    lat_min, lon_min = 19.11, 72.85
    lat_max, lon_max = 19.13, 72.87
    
    # Fetch buildings
    results = service.get_buildings_for_area(
        lat_min=lat_min,
        lon_min=lon_min,
        lat_max=lat_max,
        lon_max=lon_max,
        confidence_threshold=0.75
    )
    
    # Save results
    service.save_results(results, 'outputs/gee_buildings/anand_nagar_buildings.json')
    service.export_to_geojson(results, 'outputs/gee_buildings/anand_nagar_buildings.geojson')
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total buildings: {results['total_buildings']:,}")
    print(f"Area: {results['area_km2']:.2f} km²")
    print(f"Density: {results['density_per_km2']:.1f} buildings/km²")
    
    if results['total_buildings'] > 0:
        print(f"Total building area: {results['statistics']['total_building_area_m2']:,.0f} m²")
        print(f"Average building size: {results['statistics']['average_building_size_m2']:.0f} m²")
        print(f"Confidence range: {results['statistics']['confidence']['min']:.2f} - {results['statistics']['confidence']['max']:.2f}")
    
    print("="*70)
    print("\n✅ GEE BUILDING SERVICE TEST COMPLETE!")


if __name__ == "__main__":
    main()
