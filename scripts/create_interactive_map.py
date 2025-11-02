"""
Interactive HTML Map Visualization for Encroachment Detection
Creates professional Leaflet-based interactive map with before/after slider
MUCH better than static images!
"""

import folium
from folium import plugins
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image


class InteractiveMapVisualizer:
    """Create interactive HTML maps for encroachment detection"""
    
    def __init__(self):
        self.colors = {
            'high': '#dc2626',      # Red for high severity
            'medium': '#f59e0b',    # Orange for medium
            'low': '#10b981',       # Green for low
            'change': '#ef4444',    # Bright red for changes
            'building': '#3b82f6'   # Blue for buildings
        }
    
    def create_interactive_comparison(self, change_mask, t0_path, t1_path, 
                                     aoi_bounds, start_date, end_date, 
                                     gee_analysis=None):
        """
        Create interactive map with before/after image slider
        THIS IS THE PROFESSIONAL VERSION!
        """
        
        print("\n" + "="*70)
        print("🗺️  CREATING INTERACTIVE HTML MAP")
        print("="*70)
        
        # Calculate center point
        lon_min, lat_min, lon_max, lat_max = aoi_bounds
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles=None,  # We'll add custom tiles
            control_scale=True
        )
        
        # Add multiple basemap options
        folium.TileLayer(
            'OpenStreetMap',
            name='Street Map',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add change detection overlay
        self._add_change_overlay(m, change_mask, aoi_bounds, t1_path)
        
        # Add detected regions with popups
        if gee_analysis and 'analyzed_regions' in gee_analysis:
            self._add_region_markers(m, gee_analysis, aoi_bounds, change_mask.shape)
        
        # Add before/after image comparison (side-by-side images in popup)
        self._add_comparison_control(m, t0_path, t1_path, start_date, end_date)
        
        # Add statistics panel
        self._add_statistics_panel(m, change_mask, gee_analysis, start_date, end_date)
        
        # Add measurement tool
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='meters',
            secondary_length_unit='kilometers',
            primary_area_unit='sqmeters',
            secondary_area_unit='hectares'
        ).add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen(
            position='topleft',
            title='Fullscreen',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Add minimap
        minimap = plugins.MiniMap(
            tile_layer='OpenStreetMap',
            position='bottomleft'
        )
        m.add_child(minimap)
        
        # Save
        output_path = Path("outputs/interactive_map.html")
        m.save(str(output_path))
        
        print(f"✅ Interactive map saved: {output_path}")
        print(f"🌐 Open in browser: file:///{output_path.absolute()}")
        
        return output_path
    
    def _add_change_overlay(self, map_obj, change_mask, aoi_bounds, t1_path):
        """Add semi-transparent change detection overlay"""
        
        # Load T1 image
        img = cv2.imread(str(t1_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ensure same size
        h, w = change_mask.shape[:2]
        img_rgb = cv2.resize(img_rgb, (w, h))
        
        # Create red overlay for changes
        overlay = img_rgb.copy()
        overlay[change_mask > 0] = [239, 68, 68]  # Red
        
        # Blend
        result = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)
        
        # Convert to base64 for embedding
        pil_img = Image.fromarray(result)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Add as image overlay
        lon_min, lat_min, lon_max, lat_max = aoi_bounds
        
        image_overlay = folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_str}",
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.7,
            name='Change Detection Overlay',
            overlay=True,
            control=True
        )
        image_overlay.add_to(map_obj)
    
    def _add_region_markers(self, map_obj, gee_analysis, aoi_bounds, img_shape):
        """Add markers for detected change regions"""
        
        lon_min, lat_min, lon_max, lat_max = aoi_bounds
        h, w = img_shape
        
        for region in gee_analysis['analyzed_regions']:
            # Convert pixel bbox to lat/lon
            bbox = region['bbox']
            x, y, bw, bh = bbox
            
            # Calculate center
            center_x = x + bw/2
            center_y = y + bh/2
            
            # Convert to lat/lon
            center_lon = lon_min + (center_x / w) * (lon_max - lon_min)
            center_lat = lat_max - (center_y / h) * (lat_max - lat_min)
            
            # Create detailed popup
            popup_html = self._create_region_popup(region)
            
            # Determine icon color based on severity
            buildings_new = region['buildings']['new']
            area_new = abs(region['area_m2']['new'])  # FIX: Use absolute value
            
            if buildings_new > 0:
                if buildings_new >= 10 or area_new > 5000:
                    icon_color = 'red'
                    icon_icon = 'exclamation-triangle'
                elif buildings_new >= 5 or area_new > 1000:
                    icon_color = 'orange'
                    icon_icon = 'warning'
                else:
                    icon_color = 'blue'
                    icon_icon = 'info-sign'
            else:
                icon_color = 'gray'
                icon_icon = 'remove'
            
            # Add marker
            folium.Marker(
                location=[center_lat, center_lon],
                popup=folium.Popup(popup_html, max_width=400),
                tooltip=f"Region #{region['region_id']}: {buildings_new:+d} buildings",
                icon=folium.Icon(color=icon_color, icon=icon_icon, prefix='glyphicon')
            ).add_to(map_obj)
            
            # Add polygon outline
            region_bounds = self._bbox_to_latlon(bbox, aoi_bounds, img_shape)
            
            folium.Rectangle(
                bounds=region_bounds,
                color='#fbbf24' if buildings_new > 0 else '#6b7280',
                fill=True,
                fillColor='#fef3c7' if buildings_new > 0 else '#e5e7eb',
                fillOpacity=0.3,
                weight=3,
                popup=folium.Popup(f"Region #{region['region_id']}", max_width=200)
            ).add_to(map_obj)
    
    def _bbox_to_latlon(self, bbox, aoi_bounds, img_shape):
        """Convert pixel bbox to lat/lon bounds"""
        
        lon_min, lat_min, lon_max, lat_max = aoi_bounds
        h, w = img_shape
        
        x, y, bw, bh = bbox
        
        # Convert corners
        sw_lon = lon_min + (x / w) * (lon_max - lon_min)
        sw_lat = lat_max - ((y + bh) / h) * (lat_max - lat_min)
        
        ne_lon = lon_min + ((x + bw) / w) * (lon_max - lon_min)
        ne_lat = lat_max - (y / h) * (lat_max - lat_min)
        
        return [[sw_lat, sw_lon], [ne_lat, ne_lon]]
    
    def _create_region_popup(self, region):
        """Create detailed HTML popup for a region"""
        
        buildings_before = region['buildings']['before']
        buildings_after = region['buildings']['after']
        buildings_new = region['buildings']['new']
        
        area_before = region['area_m2']['before']
        area_after = region['area_m2']['after']
        area_new = region['area_m2']['new']
        
        # FIX: Handle negative values properly
        area_new_abs = abs(area_new)
        change_direction = "added" if area_new >= 0 else "removed"
        
        html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 300px;">
            <h4 style="margin: 0 0 10px 0; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 5px;">
                📍 Region #{region['region_id']}
            </h4>
            
            <div style="background: #f9fafb; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <p style="margin: 5px 0; font-size: 13px;">
                    <strong>Type:</strong> {region['encroachment_type']}
                </p>
                <p style="margin: 5px 0; font-size: 13px;">
                    <strong>Description:</strong><br>{region['description']}
                </p>
            </div>
            
            <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                <tr style="background: #f3f4f6;">
                    <th style="padding: 5px; text-align: left; border: 1px solid #e5e7eb;">Metric</th>
                    <th style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">Before</th>
                    <th style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">After</th>
                    <th style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">Change</th>
                </tr>
                <tr>
                    <td style="padding: 5px; border: 1px solid #e5e7eb;">Buildings</td>
                    <td style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">{buildings_before}</td>
                    <td style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">{buildings_after}</td>
                    <td style="padding: 5px; text-align: right; border: 1px solid #e5e7eb; font-weight: bold; color: {'#dc2626' if buildings_new > 0 else '#6b7280'};">
                        {buildings_new:+d}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 5px; border: 1px solid #e5e7eb;">Area (m²)</td>
                    <td style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">{area_before:,.0f}</td>
                    <td style="padding: 5px; text-align: right; border: 1px solid #e5e7eb;">{area_after:,.0f}</td>
                    <td style="padding: 5px; text-align: right; border: 1px solid #e5e7eb; font-weight: bold; color: {'#dc2626' if area_new > 0 else '#6b7280'};">
                        {area_new:+,.0f}
                    </td>
                </tr>
            </table>
            
            <div style="margin-top: 10px; padding: 8px; background: {'#fef2f2' if buildings_new > 0 else '#f9fafb'}; border-left: 4px solid {'#dc2626' if buildings_new > 0 else '#6b7280'}; border-radius: 3px;">
                <p style="margin: 0; font-size: 12px;">
                    <strong>{area_new_abs:,.0f} m²</strong> of area <strong>{change_direction}</strong>
                </p>
            </div>
        </div>
        """
        
        return html
    
    def _add_comparison_control(self, map_obj, t0_path, t1_path, start_date, end_date):
        """Add before/after image comparison in legend"""
        
        # Create thumbnail comparison
        img_before = cv2.imread(str(t0_path))
        img_after = cv2.imread(str(t1_path))
        
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        # Resize to thumbnail
        thumbnail_size = (200, 200)
        img_before = cv2.resize(img_before, thumbnail_size)
        img_after = cv2.resize(img_after, thumbnail_size)
        
        # Convert to base64
        def img_to_base64(img):
            pil_img = Image.fromarray(img)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        before_b64 = img_to_base64(img_before)
        after_b64 = img_to_base64(img_after)
        
        # Create legend HTML
        legend_html = f"""
        <div style="position: fixed; bottom: 50px; right: 20px; width: 220px; background: white; 
                    border: 2px solid #e5e7eb; border-radius: 8px; padding: 10px; z-index: 9999;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin: 0 0 10px 0; font-size: 14px; color: #1f2937;">Image Comparison</h4>
            
            <div style="margin-bottom: 10px;">
                <p style="margin: 5px 0; font-size: 11px; font-weight: bold;">T₀: Before</p>
                <img src="data:image/png;base64,{before_b64}" style="width: 100%; border-radius: 4px;">
                <p style="margin: 2px 0; font-size: 10px; color: #6b7280;">{self._format_date(start_date)}</p>
            </div>
            
            <div>
                <p style="margin: 5px 0; font-size: 11px; font-weight: bold;">T₁: After</p>
                <img src="data:image/png;base64,{after_b64}" style="width: 100%; border-radius: 4px;">
                <p style="margin: 2px 0; font-size: 10px; color: #6b7280;">{self._format_date(end_date)}</p>
            </div>
            
            <div style="margin-top: 10px; padding: 8px; background: #fef2f2; border-radius: 4px;">
                <p style="margin: 0; font-size: 11px; color: #dc2626; font-weight: bold;">
                    🔴 Red = Changes Detected
                </p>
            </div>
        </div>
        """
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_statistics_panel(self, map_obj, change_mask, gee_analysis, start_date, end_date):
        """Add statistics panel"""
        
        total_pixels = change_mask.size
        changed_pixels = int(np.sum(change_mask > 0))
        change_pct = (changed_pixels / total_pixels) * 100
        
        # Get GEE stats if available
        if gee_analysis and 'analyzed_regions' in gee_analysis:
            total_regions = len(gee_analysis['analyzed_regions'])
            total_new_buildings = sum(r['buildings']['new'] for r in gee_analysis['analyzed_regions'])
            total_new_area = sum(abs(r['area_m2']['new']) for r in gee_analysis['analyzed_regions'])
        else:
            total_regions = 0
            total_new_buildings = 0
            total_new_area = 0
        
        stats_html = f"""
        <div style="position: fixed; top: 80px; right: 20px; width: 280px; background: white;
                    border: 2px solid #e5e7eb; border-radius: 8px; padding: 15px; z-index: 9999;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">
                📊 Detection Statistics
            </h3>
            
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; font-size: 11px; color: #6b7280; text-transform: uppercase;">Analysis Period</p>
                <p style="margin: 2px 0; font-size: 13px; font-weight: bold; color: #1f2937;">
                    {self._format_date(start_date)}<br>→ {self._format_date(end_date)}
                </p>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                <div style="background: #f9fafb; padding: 8px; border-radius: 4px;">
                    <p style="margin: 0; font-size: 10px; color: #6b7280;">Change %</p>
                    <p style="margin: 2px 0; font-size: 18px; font-weight: bold; color: #dc2626;">{change_pct:.2f}%</p>
                </div>
                <div style="background: #f9fafb; padding: 8px; border-radius: 4px;">
                    <p style="margin: 0; font-size: 10px; color: #6b7280;">Regions</p>
                    <p style="margin: 2px 0; font-size: 18px; font-weight: bold; color: #3b82f6;">{total_regions}</p>
                </div>
            </div>
            
            <div style="background: #fef2f2; padding: 10px; border-radius: 4px; border-left: 4px solid #dc2626;">
                <p style="margin: 0 0 5px 0; font-size: 12px; font-weight: bold; color: #dc2626;">
                    GEE Building Analysis
                </p>
                <p style="margin: 3px 0; font-size: 11px; color: #1f2937;">
                    <strong>{total_new_buildings:+d}</strong> buildings detected
                </p>
                <p style="margin: 3px 0; font-size: 11px; color: #1f2937;">
                    <strong>{total_new_area:,.0f} m²</strong> total change
                </p>
            </div>
            
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 10px; color: #6b7280; text-align: center;">
                    Powered by Google Earth Engine & Sentinel-2
                </p>
            </div>
        </div>
        """
        
        map_obj.get_root().html.add_child(folium.Element(stats_html))
    
    def _format_date(self, date_str):
        """Format date string"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%b %d, %Y")
        except:
            return date_str


def main():
    """Generate interactive HTML map"""
    
    print("\n" + "="*70)
    print("🗺️  INTERACTIVE MAP GENERATOR")
    print("="*70)
    
    # Load data
    raw_dir = Path("data/raw")
    t0_path = raw_dir / "t0.png"
    t1_path = raw_dir / "t1.png"
    
    if not t0_path.exists():
        print("\n❌ Run pipeline first!")
        return
    
    # Load change mask
    print("\n📊 Loading detection results...")
    from run_pipeline import preprocess_image, run_inference
    
    img_t0 = cv2.imread(str(t0_path))
    img_t1 = cv2.imread(str(t1_path))
    
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    
    change_mask = run_inference(t0_norm, t1_norm)
    
    # Load GEE analysis if available
    gee_analysis = None
    gee_file = Path("outputs/gee_enhanced_analysis.json")
    if gee_file.exists():
        print("📊 Loading GEE analysis...")
        with open(gee_file, 'r') as f:
            gee_analysis = json.load(f)
    
    # Parameters
    aoi_bounds = [72.85, 19.05, 72.90, 19.15]  # Update with your actual bounds
    start_date = "2023-09-01"
    end_date = "2024-03-01"
    
    # Create visualizer
    visualizer = InteractiveMapVisualizer()
    
    # Generate interactive map
    output_path = visualizer.create_interactive_comparison(
        change_mask, str(t0_path), str(t1_path),
        aoi_bounds, start_date, end_date, gee_analysis
    )
    
    print("\n" + "="*70)
    print("✅ INTERACTIVE MAP COMPLETE!")
    print("="*70)
    print(f"\n📂 Open this file in your browser:")
    print(f"   {output_path.absolute()}")
    print(f"\n🎯 Features:")
    print(f"   ✅ Zoomable/pannable map")
    print(f"   ✅ Multiple basemaps (Street/Satellite)")
    print(f"   ✅ Interactive region markers with detailed popups")
    print(f"   ✅ Before/after image comparison")
    print(f"   ✅ Statistics panel")
    print(f"   ✅ Measurement tools")
    print(f"   ✅ Fullscreen mode")
    print("\n🎉 Perfect for Thursday demo!")


if __name__ == "__main__":
    main()
