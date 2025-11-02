"""
Batch GEE building detection for multiple Mumbai areas
"""

from gee_building_service import GEEBuildingService
from pathlib import Path
import json

# Mumbai areas for your Thursday demo
DEMO_AREAS = {
    'anand_nagar': {
        'name': 'Anand Nagar',
        'bounds': (19.11, 72.85, 19.13, 72.87)
    },
    'bkc': {
        'name': 'BKC',
        'bounds': (19.05, 72.86, 19.08, 72.88)
    },
    'worli': {
        'name': 'Worli',
        'bounds': (19.00, 72.81, 19.03, 72.83)
    }
}

def main():
    print("\n" + "="*70)
    print("🚀 BATCH GEE BUILDING DETECTION - MUMBAI DEMO AREAS")
    print("="*70)
    
    # Initialize GEE
    service = GEEBuildingService(project_id='urbaneye-476904')
    
    results_summary = []
    
    # Process each area
    for area_id, area_info in DEMO_AREAS.items():
        print(f"\n📍 Processing: {area_info['name']}")
        print("-" * 70)
        
        lat_min, lon_min, lat_max, lon_max = area_info['bounds']
        
        # Fetch buildings
        results = service.get_buildings_for_area(
            lat_min=lat_min,
            lon_min=lon_min,
            lat_max=lat_max,
            lon_max=lon_max,
            confidence_threshold=0.75
        )
        
        # Save results
        output_dir = Path(f'outputs/gee_buildings/{area_id}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        service.save_results(results, output_dir / 'buildings.json')
        service.export_to_geojson(results, output_dir / 'buildings.geojson')
        
        # Add to summary
        results_summary.append({
            'area_id': area_id,
            'name': area_info['name'],
            'total_buildings': results['total_buildings'],
            'area_km2': results['area_km2'],
            'density_per_km2': results['density_per_km2'],
            'avg_building_size_m2': results['statistics']['average_building_size_m2'] if results['total_buildings'] > 0 else 0
        })
    
    # Print summary table
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"{'Area':<20} {'Buildings':>12} {'Density':>12} {'Avg Size':>12}")
    print("-" * 70)
    
    for item in results_summary:
        print(f"{item['name']:<20} {item['total_buildings']:>12,} "
              f"{item['density_per_km2']:>11.1f}/km² "
              f"{item['avg_building_size_m2']:>11.0f} m²")
    
    print("="*70)
    
    # Save summary
    with open('outputs/gee_buildings/batch_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✅ Batch processing complete!")
    print("💾 Results saved to: outputs/gee_buildings/")
    print("\n🎯 Ready for your Thursday demo!")

if __name__ == "__main__":
    main()
