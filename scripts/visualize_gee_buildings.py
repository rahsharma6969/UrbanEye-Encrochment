"""
Visualize GEE building data on an interactive map
"""

import json
import folium
from pathlib import Path

def load_geojson(file_path):
    """Load GeoJSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_building_map(geojson_path, output_html='outputs/building_map.html'):
    """
    Create interactive map with building footprints
    
    Args:
        geojson_path: Path to GeoJSON file
        output_html: Output HTML file path
    """
    
    print(f"\n{'='*70}")
    print(f"🗺️  CREATING BUILDING MAP")
    print(f"{'='*70}")
    print(f"📂 Loading: {geojson_path}")
    
    # Load GeoJSON
    data = load_geojson(geojson_path)
    
    # Get metadata
    metadata = data.get('metadata', {})
    total_buildings = metadata.get('total_buildings', 0)
    area_km2 = metadata.get('area_km2', 0)
    density = metadata.get('density_per_km2', 0)
    bounds = metadata.get('bounds', [72.85, 19.11, 72.87, 19.13])
    
    print(f"🏢 Total buildings: {total_buildings:,}")
    print(f"📊 Density: {density:.1f} buildings/km²")
    print(f"📐 Area: {area_km2:.2f} km²")
    
    # Calculate center
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery option
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color buildings by confidence
    def get_color(confidence):
        """Green = high confidence, Yellow = medium, Red = low"""
        if confidence >= 0.85:
            return '#00ff00'  # Green
        elif confidence >= 0.75:
            return '#ffff00'  # Yellow
        else:
            return '#ff0000'  # Red
    
    # Add buildings to map
    print(f"⏳ Adding {len(data['features'])} buildings to map...")
    
    feature_group = folium.FeatureGroup(name='Buildings (Google Open Buildings)')
    
    for idx, feature in enumerate(data['features']):
        if idx % 500 == 0:
            print(f"   Processing {idx}/{len(data['features'])}...")
        
        props = feature.get('properties', {})
        confidence = props.get('confidence', 0.7)
        area_sqm = props.get('area_in_meters', props.get('area_sqm', 0))
        
        # Create popup
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Building #{idx + 1}</b><br>
            <b>Confidence:</b> {confidence:.2%}<br>
            <b>Area:</b> {area_sqm:.1f} m²<br>
            <b>Source:</b> Google Open Buildings
        </div>
        """
        
        # Add polygon
        folium.GeoJson(
            feature,
            style_function=lambda x, conf=confidence: {
                'fillColor': get_color(conf),
                'color': get_color(conf),
                'weight': 1,
                'fillOpacity': 0.5
            },
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(feature_group)
    
    feature_group.add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="
        position: fixed; 
        top: 10px; right: 10px; 
        width: 250px; 
        background-color: white; 
        border: 2px solid grey; 
        z-index: 9999; 
        padding: 10px;
        font-family: Arial;
        font-size: 12px;
        border-radius: 5px;
    ">
        <h4 style="margin-top: 0;">🏗️ Building Detection</h4>
        <p><b>Total Buildings:</b> {total_buildings:,}</p>
        <p><b>Area:</b> {area_km2:.2f} km²</p>
        <p><b>Density:</b> {density:.1f}/km²</p>
        <hr>
        <p><b>Confidence Levels:</b></p>
        <p><span style="color: #00ff00;">●</span> High (≥85%)</p>
        <p><span style="color: #ffff00;">●</span> Medium (75-85%)</p>
        <p><span style="color: #ff0000;">●</span> Low (<75%)</p>
        <hr>
        <p style="font-size: 10px; color: grey;">
            Data: Google Open Buildings<br>
            Generated: {metadata.get('generated_at', 'N/A')}
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    print(f"\n✅ Map created successfully!")
    print(f"💾 Saved to: {output_path}")
    print(f"🌐 Open in browser: file:///{output_path.absolute()}")
    print(f"{'='*70}\n")
    
    return str(output_path.absolute())

def create_comparison_map(output_html='outputs/comparison_map.html'):
    """Create map comparing all 3 demo areas"""
    
    print(f"\n{'='*70}")
    print(f"🗺️  CREATING COMPARISON MAP - ALL DEMO AREAS")
    print(f"{'='*70}")
    
    # Center on Mumbai
    m = folium.Map(
        location=[19.076, 72.877],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Load all areas
    areas = {
        'Anand Nagar': {
            'path': 'outputs/gee_buildings/anand_nagar/buildings.geojson',
            'color': '#FF0000'
        },
        'BKC': {
            'path': 'outputs/gee_buildings/bkc/buildings.geojson',
            'color': '#00FF00'
        },
        'Worli': {
            'path': 'outputs/gee_buildings/worli/buildings.geojson',
            'color': '#0000FF'
        }
    }
    
    summary_html = "<h4>📊 Detection Summary</h4>"
    
    for area_name, area_info in areas.items():
        path = Path(area_info['path'])
        
        if not path.exists():
            print(f"⚠️  {area_name}: File not found - {path}")
            continue
        
        print(f"📍 Loading {area_name}...")
        
        data = load_geojson(path)
        metadata = data.get('metadata', {})
        
        # Create feature group for this area
        fg = folium.FeatureGroup(name=f'{area_name} ({metadata.get("total_buildings", 0):,} buildings)')
        
        # Add buildings (sample for performance)
        sample_size = min(500, len(data['features']))
        for feature in data['features'][:sample_size]:
            folium.GeoJson(
                feature,
                style_function=lambda x, color=area_info['color']: {
                    'fillColor': color,
                    'color': color,
                    'weight': 1,
                    'fillOpacity': 0.4
                }
            ).add_to(fg)
        
        fg.add_to(m)
        
        # Add to summary
        summary_html += f"""
        <p style="color: {area_info['color']};">
            <b>{area_name}:</b> {metadata.get('total_buildings', 0):,} buildings<br>
            Density: {metadata.get('density_per_km2', 0):.1f}/km²
        </p>
        """
    
    # Add legend
    legend_html = f"""
    <div style="
        position: fixed; 
        top: 10px; right: 10px; 
        width: 280px; 
        background-color: white; 
        border: 2px solid grey; 
        z-index: 9999; 
        padding: 10px;
        font-family: Arial;
        font-size: 12px;
        border-radius: 5px;
    ">
        {summary_html}
        <hr>
        <p style="font-size: 10px; color: grey;">
            Data: Google Open Buildings V3<br>
            Coverage: Mumbai, India
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save
    output_path = Path(output_html)
    m.save(str(output_path))
    
    print(f"\n✅ Comparison map created!")
    print(f"💾 Saved to: {output_path}")
    print(f"🌐 Open in browser: file:///{output_path.absolute()}")
    print(f"{'='*70}\n")

def main():
    """Create all visualizations"""
    
    print("\n" + "="*70)
    print("🎨 GEE BUILDING VISUALIZATION TOOL")
    print("="*70)
    
    # Create individual maps
    areas = [
        ('outputs/gee_buildings/anand_nagar/buildings.geojson', 'outputs/maps/anand_nagar_map.html'),
        ('outputs/gee_buildings/bkc/buildings.geojson', 'outputs/maps/bkc_map.html'),
        ('outputs/gee_buildings/worli/buildings.geojson', 'outputs/maps/worli_map.html')
    ]
    
    for geojson_path, output_html in areas:
        if Path(geojson_path).exists():
            create_building_map(geojson_path, output_html)
    
    # Create comparison map
    create_comparison_map('outputs/maps/all_areas_comparison.html')
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📍 outputs/maps/anand_nagar_map.html")
    print("  📍 outputs/maps/bkc_map.html")
    print("  📍 outputs/maps/worli_map.html")
    print("  📊 outputs/maps/all_areas_comparison.html")
    print("\n🌐 Open any .html file in your browser to view!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
