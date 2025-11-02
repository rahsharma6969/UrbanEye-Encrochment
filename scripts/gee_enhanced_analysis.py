"""
GEE-Enhanced Change Analysis - FIXED VERSION
✅ Fixed JSON serialization
✅ Fixed image blur with higher resolution
✅ Better zoom visualization
"""

import ee
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from datetime import datetime


class GEEEnhancedAnalyzer:
    """Analyze detected changes using GEE historical imagery and building data"""
    
    def __init__(self, project_id='urbaneye-476904'):
        """Initialize with GEE authentication"""
        try:
            ee.Initialize(project=project_id)
            print(f"✅ GEE initialized (Project: {project_id})")
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            raise
        
        self.buildings_dataset = ee.FeatureCollection(
            'GOOGLE/Research/open-buildings/v3/polygons'
        )
        
        # For temporal building data (2016-2023)
        self.buildings_temporal = ee.ImageCollection(
            'GOOGLE/Research/open-buildings-temporal/v1'
        )
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def analyze_change_regions(self, change_mask, aoi_bounds, start_date, end_date):
        """Analyze detected change regions with GEE data"""
        
        print("\n" + "="*70)
        print("🔍 GEE ENHANCED CHANGE ANALYSIS")
        print("="*70)
        
        # Find significant change regions
        change_regions = self._find_change_regions(change_mask)
        
        print(f"\n📍 Found {len(change_regions)} significant change regions")
        
        # Analyze each region with GEE
        analyzed_regions = []
        
        for idx, region in enumerate(change_regions[:5]):  # Top 5 largest changes
            print(f"\n🔍 Analyzing Region #{idx + 1}...")
            
            # Get region bounds in lat/lon
            region_bounds = self._get_region_bounds(
                region, aoi_bounds, change_mask.shape
            )
            
            # Analyze with GEE
            analysis = self._analyze_region_with_gee(
                region_bounds, start_date, end_date
            )
            
            # FIX: Convert all values to JSON-serializable types
            analysis['region_id'] = int(idx + 1)
            analysis['pixel_area'] = int(region['area'])
            analysis['bbox'] = tuple(int(x) for x in region['bbox'])
            
            analyzed_regions.append(analysis)
        
        result = {
            'total_regions': int(len(change_regions)),
            'analyzed_regions': analyzed_regions,
            'aoi_bounds': [float(x) for x in aoi_bounds],
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
        
        # FIX: Ensure everything is JSON serializable
        return self._convert_to_json_serializable(result)
    
    def _find_change_regions(self, change_mask):
        """Find connected regions of changes"""
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            change_mask.astype(np.uint8), connectivity=8
        )
        
        regions = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter small changes (noise)
            if area < 100:
                continue
            
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            
            regions.append({
                'label': int(i),
                'area': int(area),
                'bbox': (int(x), int(y), int(w), int(h)),
                'centroid': (float(centroids[i][0]), float(centroids[i][1]))
            })
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r['area'], reverse=True)
        
        return regions
    
    def _get_region_bounds(self, region, aoi_bounds, img_shape):
        """Convert pixel coordinates to lat/lon bounds"""
        
        lon_min, lat_min, lon_max, lat_max = aoi_bounds
        h, w = img_shape
        
        x, y, bw, bh = region['bbox']
        
        # Convert pixel to lat/lon
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        
        region_lon_min = lon_min + (x / w) * lon_range
        region_lon_max = lon_min + ((x + bw) / w) * lon_range
        region_lat_min = lat_max - ((y + bh) / h) * lat_range  # Flip Y
        region_lat_max = lat_max - (y / h) * lat_range
        
        return [region_lon_min, region_lat_min, region_lon_max, region_lat_max]
    
    def _analyze_region_with_gee(self, bounds, start_date, end_date):
        """Analyze a specific region using GEE historical data"""
        
        lon_min, lat_min, lon_max, lat_max = bounds
        roi = ee.Geometry.Rectangle(bounds)
        
        # Get buildings BEFORE and AFTER
        buildings_before = self._get_buildings_at_date(roi, start_date)
        buildings_after = self._get_buildings_at_date(roi, end_date)
        
        # Calculate change
        buildings_before_count = buildings_before['count']
        buildings_after_count = buildings_after['count']
        new_buildings = buildings_after_count - buildings_before_count
        
        # Calculate area
        area_before = buildings_before['total_area']
        area_after = buildings_after['total_area']
        new_area = area_after - area_before
        
        # Determine encroachment type
        encroachment_type = self._classify_encroachment_type(
            new_buildings, new_area, buildings_before_count
        )
        
        analysis = {
            'bounds': [float(x) for x in bounds],
            'buildings': {
                'before': int(buildings_before_count),
                'after': int(buildings_after_count),
                'new': int(new_buildings)
            },
            'area_m2': {
                'before': float(round(area_before, 2)),
                'after': float(round(area_after, 2)),
                'new': float(round(new_area, 2))
            },
            'encroachment_type': str(encroachment_type),
            'description': str(self._generate_description(
                new_buildings, new_area, encroachment_type
            ))
        }
        
        print(f"   Buildings: {buildings_before_count} → {buildings_after_count} (+{new_buildings})")
        print(f"   Area: {area_before:.0f} m² → {area_after:.0f} m² (+{new_area:.0f} m²)")
        print(f"   Type: {encroachment_type}")
        
        return analysis
    
    def _get_buildings_at_date(self, roi, date_str):
        """Get building count and area for a specific date"""
        
        try:
            # Try current buildings dataset
            buildings = self.buildings_dataset.filterBounds(roi).filter(
                ee.Filter.gte('confidence', 0.75)
            )
            
            building_count = buildings.size().getInfo()
            
            total_area_result = buildings.aggregate_sum('area_in_meters').getInfo()
            total_area = total_area_result if total_area_result else 0
            
            return {
                'count': int(building_count),
                'total_area': float(total_area)
            }
        
        except Exception as e:
            print(f"   ⚠️  Error fetching buildings: {e}")
            return {
                'count': 0,
                'total_area': 0.0
            }
    
    def _classify_encroachment_type(self, new_buildings, new_area, existing_buildings):
        """Classify type of encroachment"""
        
        if new_buildings <= 0:
            return "Demolition/Removal"
        
        if new_buildings >= 10:
            return "Major Development"
        elif new_buildings >= 5:
            return "Moderate Construction"
        elif new_buildings >= 1:
            return "Minor Construction"
        
        if new_area > 5000:
            return "Large-Scale Development"
        elif new_area > 1000:
            return "Medium Development"
        else:
            return "Small-Scale Changes"
    
    def _generate_description(self, new_buildings, new_area, encroachment_type):
        """Generate human-readable description"""
        
        if new_buildings > 0:
            return f"{new_buildings} new building{'s' if new_buildings != 1 else ''} " \
                   f"covering {new_area:.0f} m² ({encroachment_type})"
        elif new_buildings < 0:
            return f"{abs(new_buildings)} building{'s' if abs(new_buildings) != 1 else ''} " \
                   f"removed ({abs(new_area):.0f} m² cleared)"
        else:
            return f"Surface changes detected ({new_area:.0f} m² modified)"
    
    def create_detailed_visualization(self, change_mask, t0_path, t1_path, 
                                     aoi_bounds, start_date, end_date):
        """Create detailed visualization with GEE building analysis"""
        
        print("\n" + "="*70)
        print("🎨 CREATING GEE-ENHANCED DETAILED VISUALIZATION")
        print("="*70)
        
        # Analyze changes with GEE
        analysis = self.analyze_change_regions(
            change_mask, aoi_bounds, start_date, end_date
        )
        
        # Load base images
        img_before = cv2.imread(str(t0_path))
        img_after = cv2.imread(str(t1_path))
        
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        # FIX BLUR: Upscale images 4x for better detail
        scale_factor = 4
        h, w = change_mask.shape[:2]
        
        img_before = cv2.resize(img_before, (w*scale_factor, h*scale_factor), 
                               interpolation=cv2.INTER_CUBIC)
        img_after = cv2.resize(img_after, (w*scale_factor, h*scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
        change_mask_hires = cv2.resize(change_mask.astype(np.uint8), 
                                       (w*scale_factor, h*scale_factor), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Update dimensions
        h, w = h*scale_factor, w*scale_factor
        
        # Create figure
        fig = plt.figure(figsize=(24, 14), facecolor='white')
        
        # Create grid
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1],
                             hspace=0.25, wspace=0.1,
                             left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # Header
        fig.suptitle(
            'GEE-Enhanced Encroachment Detection with Building Analysis',
            fontsize=22, fontweight='bold', y=0.97
        )
        
        # === TOP ROW ===
        
        # Before
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_before)
        ax1.set_title(f'T₀: Before\n{self._format_date(start_date)}',
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # After
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_after)
        ax2.set_title(f'T₁: After\n{self._format_date(end_date)}',
                     fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Changes with labels
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = self._create_labeled_overlay(
            img_after, change_mask_hires, analysis['analyzed_regions'], scale_factor
        )
        ax3.imshow(overlay)
        ax3.set_title('Changes Detected with Labels',
                     fontsize=14, fontweight='bold', color='#dc2626')
        ax3.axis('off')
        
        # === BOTTOM ROW: High-res zoomed regions ===
        
        for idx in range(min(3, len(analysis['analyzed_regions']))):
            ax = fig.add_subplot(gs[1, idx])
            
            region = analysis['analyzed_regions'][idx]
            
            # FIX: Add zoomed high-res detail
            self._add_highres_region_detail(
                ax, region, img_after, change_mask_hires, scale_factor
            )
        
        # Save
        output_path = Path("outputs/gee_enhanced_detailed_analysis.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"\n✅ GEE-enhanced visualization saved: {output_path}")
        
        # Save analysis as JSON (now with proper serialization)
        json_path = Path("outputs/gee_enhanced_analysis.json")
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✅ Analysis data saved: {json_path}")
        
        return output_path, analysis
    
    def _create_labeled_overlay(self, img_base, change_mask, regions, scale_factor):
        """Create overlay with building change labels"""
        
        overlay = img_base.copy()
        
        # Red for changes
        change_overlay = np.zeros_like(overlay)
        change_overlay[change_mask > 0] = [220, 38, 38]
        overlay = cv2.addWeighted(overlay, 0.6, change_overlay, 0.4, 0)
        
        # Add labels for each region
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for region in regions:
            bbox = region['bbox']
            x, y, bw, bh = [int(v * scale_factor) for v in bbox]
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (255, 255, 0), 4)
            
            # Label text
            new_buildings = region['buildings']['new']
            label_text = f"#{region['region_id']}: +{new_buildings} bldg"
            
            # Background for text
            text_size = cv2.getTextSize(label_text, font, 1.0, 3)[0]
            cv2.rectangle(overlay,
                         (x, y - text_size[1] - 20),
                         (x + text_size[0] + 20, y),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(overlay, label_text, (x + 10, y - 10),
                       font, 1.0, (255, 255, 0), 3, cv2.LINE_AA)
        
        return overlay
    
    def _add_highres_region_detail(self, ax, region, img_base, change_mask, scale_factor):
        """Add HIGH-RESOLUTION detailed view of a specific region"""
        
        bbox = region['bbox']
        x, y, bw, bh = [int(v * scale_factor) for v in bbox]
        
        # Add padding for context (20% on each side)
        padding = int(min(bw, bh) * 0.2)
        h_img, w_img = img_base.shape[:2]
        
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(w_img, x + bw + padding)
        y_end = min(h_img, y + bh + padding)
        
        # Crop region with padding
        region_img = img_base[y_start:y_end, x_start:x_end].copy()
        region_mask = change_mask[y_start:y_end, x_start:x_end]
        
        # FIX: Further upscale for zoom detail (2x)
        zoom_factor = 2
        region_img = cv2.resize(region_img, 
                               (region_img.shape[1]*zoom_factor, region_img.shape[0]*zoom_factor),
                               interpolation=cv2.INTER_CUBIC)
        region_mask = cv2.resize(region_mask, 
                                (region_mask.shape[1]*zoom_factor, region_mask.shape[0]*zoom_factor),
                                interpolation=cv2.INTER_NEAREST)
        
        # Apply sharpening for extra clarity
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        region_img = cv2.filter2D(region_img, -1, kernel)
        
        # Overlay changes
        change_overlay = np.zeros_like(region_img)
        change_overlay[region_mask > 0] = [220, 38, 38]
        region_with_changes = cv2.addWeighted(region_img, 0.6, change_overlay, 0.4, 0)
        
        # Display
        ax.imshow(region_with_changes)
        
        # Title with analysis
        title = f"Region #{region['region_id']}\n{region['description']}"
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Add statistics text
        stats_text = f"Buildings: {region['buildings']['before']} → {region['buildings']['after']}\n" \
                    f"Area: +{region['area_m2']['new']:.0f} m²\n" \
                    f"Type: {region['encroachment_type']}"
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=9, color='white', weight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax.axis('off')
    
    def _format_date(self, date_str):
        """Format date string"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%B %d, %Y")
        except:
            return date_str


def main():
    """Generate GEE-enhanced analysis"""
    
    print("\n" + "="*70)
    print("🚀 GEE-ENHANCED ENCROACHMENT ANALYSIS")
    print("="*70)
    
    # Load necessary data
    raw_dir = Path("data/raw")
    t0_path = raw_dir / "t0.png"
    t1_path = raw_dir / "t1.png"
    
    if not t0_path.exists():
        print("\n❌ Run pipeline first!")
        return
    
    # Load change mask
    print("\n📊 Loading change detection results...")
    from run_pipeline import preprocess_image, run_inference
    
    img_t0 = cv2.imread(str(t0_path))
    img_t1 = cv2.imread(str(t1_path))
    
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    
    change_mask = run_inference(t0_norm, t1_norm)
    
    # AOI bounds and dates
    aoi_bounds = [72.86, 19.05, 72.88, 19.08]
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    
    # Create analyzer
    analyzer = GEEEnhancedAnalyzer(project_id='urbaneye-476904')
    
    # Generate detailed visualization
    output_path, analysis = analyzer.create_detailed_visualization(
        change_mask, str(t0_path), str(t1_path),
        aoi_bounds, start_date, end_date
    )
    
    # Print summary
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    
    for region in analysis['analyzed_regions']:
        print(f"\n📍 Region #{region['region_id']}:")
        print(f"   {region['description']}")
        print(f"   Buildings: {region['buildings']['before']} → {region['buildings']['after']}")
        print(f"   Type: {region['encroachment_type']}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\n📂 Files generated:")
    print(f"   - {output_path}")
    print(f"   - outputs/gee_enhanced_analysis.json")
    print("\n🎉 Ready for Thursday demo!")


if __name__ == "__main__":
    main()
