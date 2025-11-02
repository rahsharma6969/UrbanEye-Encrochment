"""
PROFESSIONAL-GRADE Encroachment Detection Visualization
Publication-ready output with clean design and clear information hierarchy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import json


class ProfessionalVisualizer:
    """Create publication-quality encroachment detection visualizations"""
    
    def __init__(self):
        # Color scheme - professional and accessible
        self.colors = {
            'background': (245, 245, 245),
            'header_bg': (40, 44, 52),
            'text_primary': (255, 255, 255),
            'text_secondary': (180, 180, 180),
            'change_red': (220, 38, 38),      # Bright red for changes
            'change_orange': (251, 146, 60),  # Orange for medium changes
            'change_yellow': (250, 204, 21),  # Yellow for minor changes
            'border': (200, 200, 200),
            'high_severity': (239, 68, 68),
            'medium_severity': (251, 146, 60),
            'low_severity': (34, 197, 94)
        }
    
    def create_complete_visualization(self, bin_mask, t0_path, t1_path, 
                                     start_date, end_date, classification_results=None):
        """
        Create complete professional visualization with all panels
        
        Returns:
            Path to saved visualization
        """
        
        print("\n" + "="*70)
        print("🎨 CREATING PROFESSIONAL VISUALIZATION")
        print("="*70)
        
        # Load and prepare images
        img_before, img_after, change_mask = self._prepare_images(
            t0_path, t1_path, bin_mask
        )
        
        # Create the main figure with matplotlib for professional output
        fig = self._create_professional_figure(
            img_before, img_after, change_mask,
            start_date, end_date, classification_results
        )
        
        # Save high-resolution output
        output_path = Path("outputs/professional_visualization.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
        
        print(f"✅ Professional visualization saved: {output_path}")
        print(f"   Resolution: 300 DPI (publication quality)")
        print(f"   Format: PNG with transparency")
        
        return output_path
    
    def _prepare_images(self, t0_path, t1_path, bin_mask):
        """Load and enhance images for visualization"""
        
        print("📸 Loading and preparing images...")
        
        # Load images
        img_before = cv2.imread(str(t0_path))
        img_after = cv2.imread(str(t1_path))
        
        if img_before is None or img_after is None:
            raise ValueError("Failed to load images")
        
        # Convert BGR to RGB for matplotlib
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        # Ensure same size
        h = min(img_before.shape[0], img_after.shape[0], bin_mask.shape[0])
        w = min(img_before.shape[1], img_after.shape[1], bin_mask.shape[1])
        
        img_before = cv2.resize(img_before, (w, h))
        img_after = cv2.resize(img_after, (w, h))
        bin_mask_resized = cv2.resize(bin_mask.astype(np.uint8), (w, h))
        
        # Enhance contrast
        img_before = self._enhance_for_display(img_before)
        img_after = self._enhance_for_display(img_after)
        
        print(f"   Image size: {w} x {h}")
        print(f"   Change pixels: {np.sum(bin_mask_resized > 0):,}")
        
        return img_before, img_after, bin_mask_resized
    
    def _enhance_for_display(self, img):
        """Enhance image for better display"""
        
        # Convert to float
        img_float = img.astype(np.float32) / 255.0
        
        # Increase brightness slightly
        img_float = np.clip(img_float * 1.1 + 0.05, 0, 1)
        
        # Increase contrast
        img_float = np.clip((img_float - 0.5) * 1.2 + 0.5, 0, 1)
        
        return (img_float * 255).astype(np.uint8)
    
    def _create_professional_figure(self, img_before, img_after, change_mask,
                                   start_date, end_date, classification_results):
        """Create the main professional figure using matplotlib"""
        
        print("🎨 Creating figure layout...")
        
        # Create figure with custom size (16:9 aspect ratio for presentation)
        fig = plt.figure(figsize=(20, 11), facecolor='white')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 3, 0.8], 
                             width_ratios=[1, 1, 1],
                             hspace=0.15, wspace=0.08,
                             left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # ============ HEADER ============
        ax_header = fig.add_subplot(gs[0, :])
        self._add_header(ax_header, start_date, end_date, classification_results)
        
        # ============ MAIN PANELS ============
        
        # Panel 1: Before Image
        ax_before = fig.add_subplot(gs[1, 0])
        ax_before.imshow(img_before)
        ax_before.set_title('T₀ (Before)', fontsize=16, fontweight='bold', pad=15)
        ax_before.text(0.02, 0.98, self._format_date(start_date),
                      transform=ax_before.transAxes,
                      fontsize=12, color='white', weight='bold',
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax_before.axis('off')
        
        # Panel 2: After Image
        ax_after = fig.add_subplot(gs[1, 1])
        ax_after.imshow(img_after)
        ax_after.set_title('T₁ (After)', fontsize=16, fontweight='bold', pad=15)
        ax_after.text(0.02, 0.98, self._format_date(end_date),
                     transform=ax_after.transAxes,
                     fontsize=12, color='white', weight='bold',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax_after.axis('off')
        
        # Panel 3: Changes Overlay
        ax_changes = fig.add_subplot(gs[1, 2])
        overlay = self._create_change_overlay(img_after, change_mask)
        ax_changes.imshow(overlay)
        ax_changes.set_title('Changes Detected', fontsize=16, fontweight='bold',
                            color='#dc2626', pad=15)
        
        # Add change percentage
        change_pct = (np.sum(change_mask > 0) / change_mask.size) * 100
        ax_changes.text(0.02, 0.98, f'Change: {change_pct:.2f}%',
                       transform=ax_changes.transAxes,
                       fontsize=14, color='white', weight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='#dc2626', alpha=0.9))
        ax_changes.axis('off')
        
        # ============ STATISTICS FOOTER ============
        ax_stats = fig.add_subplot(gs[2, :])
        self._add_statistics_panel(ax_stats, change_mask, classification_results)
        
        return fig
    
    def _add_header(self, ax, start_date, end_date, classification_results):
        """Add professional header with title and classification"""
        
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Main title
        ax.text(0.5, 0.7, 'UrbanEye Encroachment Detection System',
               ha='center', va='center',
               fontsize=24, fontweight='bold', color='#1f2937')
        
        # Subtitle with date range
        date_range = f'{self._format_date(start_date)} → {self._format_date(end_date)}'
        ax.text(0.5, 0.3, date_range,
               ha='center', va='center',
               fontsize=14, color='#6b7280')
        
        # Classification badge (if available)
        if classification_results:
            classification = classification_results.get('classification', 'UNKNOWN')
            severity = classification_results.get('severity', 'MEDIUM')
            confidence = classification_results.get('confidence', 0) * 100
            
            # Color based on severity
            badge_color = {
                'HIGH': '#dc2626',
                'MEDIUM': '#f59e0b',
                'LOW': '#10b981'
            }.get(severity, '#6b7280')
            
            # Classification text
            class_text = f'{classification} ({confidence:.0f}%)'
            
            bbox_props = dict(
                boxstyle='round,pad=0.5',
                facecolor=badge_color,
                edgecolor='none',
                alpha=0.9
            )
            
            ax.text(0.85, 0.5, class_text,
                   ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color='white',
                   bbox=bbox_props)
    
    def _create_change_overlay(self, img_base, change_mask):
        """Create professional change overlay with color intensity"""
        
        # Create overlay
        overlay = img_base.copy()
        
        # Apply morphological operations to clean mask
        kernel = np.ones((3,3), np.uint8)
        change_mask_clean = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        change_mask_clean = cv2.morphologyEx(change_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Create gradient overlay - darker red for solid changes
        change_overlay = np.zeros_like(overlay)
        change_overlay[change_mask_clean > 0] = [220, 38, 38]  # Red
        
        # Add yellow border around changes
        kernel_dilate = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(change_mask_clean, kernel_dilate, iterations=2)
        border = (dilated > 0) & (change_mask_clean == 0)
        change_overlay[border] = [251, 191, 36]  # Yellow border
        
        # Blend
        alpha = 0.6
        result = cv2.addWeighted(overlay, 1-alpha, change_overlay, alpha, 0)
        
        return result
    
    def _add_statistics_panel(self, ax, change_mask, classification_results):
        """Add comprehensive statistics panel"""
        
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Background
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, 
                                       facecolor='#f3f4f6', 
                                       edgecolor='#d1d5db', 
                                       linewidth=2))
        
        # Calculate statistics
        total_pixels = change_mask.size
        changed_pixels = np.sum(change_mask > 0)
        change_pct = (changed_pixels / total_pixels) * 100
        
        # Layout positions
        stats = []
        
        # Basic change statistics
        stats.append({
            'label': 'Total Pixels',
            'value': f'{total_pixels:,}',
            'x': 0.08
        })
        stats.append({
            'label': 'Changed Pixels',
            'value': f'{changed_pixels:,}',
            'x': 0.25
        })
        stats.append({
            'label': 'Change %',
            'value': f'{change_pct:.2f}%',
            'x': 0.42
        })
        
        # Add GEE statistics if available
        if classification_results and 'statistics' in classification_results:
            stats_data = classification_results['statistics']
            
            stats.append({
                'label': 'Buildings',
                'value': f"{stats_data.get('building_count', 0):,}",
                'x': 0.59
            })
            stats.append({
                'label': 'Density',
                'value': f"{stats_data.get('building_density_per_km2', 0):.1f}/km²",
                'x': 0.74
            })
            stats.append({
                'label': 'Area',
                'value': f"{stats_data.get('area_km2', 0):.2f} km²",
                'x': 0.89
            })
        
        # Draw statistics
        for stat in stats:
            # Label
            ax.text(stat['x'], 0.7, stat['label'],
                   ha='center', va='center',
                   fontsize=10, color='#6b7280')
            # Value
            ax.text(stat['x'], 0.3, stat['value'],
                   ha='center', va='center',
                   fontsize=13, fontweight='bold', color='#1f2937')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#dc2626', edgecolor='none', label='Changes Detected'),
            mpatches.Patch(facecolor='#fbbf24', edgecolor='none', label='Change Boundary')
        ]
        ax.legend(handles=legend_elements, loc='center right', 
                 frameon=False, fontsize=10)
    
    def _format_date(self, date_str):
        """Format date string to readable format"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%B %d, %Y")
        except:
            return date_str
    
    def create_comparison_heatmap(self, bin_mask, t0_path, t1_path, start_date, end_date):
        """Create heatmap-style visualization"""
        
        print("\n🔥 Creating heatmap visualization...")
        
        # Load images
        img_before = cv2.imread(str(t0_path))
        img_after = cv2.imread(str(t1_path))
        
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        # Ensure same size
        h = min(img_before.shape[0], img_after.shape[0], bin_mask.shape[0])
        w = min(img_before.shape[1], img_after.shape[1], bin_mask.shape[1])
        
        img_after = cv2.resize(img_after, (w, h))
        bin_mask_resized = cv2.resize(bin_mask.astype(np.uint8), (w, h))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        # Show base image
        ax.imshow(img_after, alpha=0.7)
        
        # Overlay heatmap
        heatmap = bin_mask_resized.astype(float)
        im = ax.imshow(heatmap, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Change Intensity', rotation=270, labelpad=20, fontsize=12)
        
        # Title
        ax.set_title(f'Change Detection Heatmap\n{self._format_date(start_date)} → {self._format_date(end_date)}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Save
        output_path = Path("outputs/change_heatmap.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✅ Heatmap saved: {output_path}")
        
        return output_path


def main():
    """Generate all professional visualizations"""
    
    print("\n" + "="*70)
    print("🎨 PROFESSIONAL VISUALIZATION GENERATOR")
    print("="*70)
    
    # Load necessary data
    raw_dir = Path("data/raw")
    t0_path = raw_dir / "t0.png"
    t1_path = raw_dir / "t1.png"
    
    if not t0_path.exists() or not t1_path.exists():
        print("\n❌ Images not found. Run pipeline first:")
        print("   python scripts/run_pipeline.py 19.05 72.86 19.08 72.88 2022-01-01 2024-01-01")
        return
    
    # Load change mask
    print("\n🔍 Running change detection...")
    from run_pipeline import preprocess_image, run_inference
    
    img_t0 = cv2.imread(str(t0_path))
    img_t1 = cv2.imread(str(t1_path))
    
    t0_norm = preprocess_image(img_t0)
    t1_norm = preprocess_image(img_t1)
    
    bin_mask = run_inference(t0_norm, t1_norm)
    
    # Load classification results if available
    classification_file = Path("outputs/classification_results.json")
    classification_results = None
    if classification_file.exists():
        print("📊 Loading classification results...")
        with open(classification_file, 'r') as f:
            classification_results = json.load(f)
    
    # Get dates
    start_date = "2022-01-01"  # Update from your actual dates
    end_date = "2024-01-01"
    
    # Create visualizations
    visualizer = ProfessionalVisualizer()
    
    print("\n" + "="*70)
    print("1️⃣  CREATING MAIN PROFESSIONAL VISUALIZATION")
    print("="*70)
    main_output = visualizer.create_complete_visualization(
        bin_mask, str(t0_path), str(t1_path),
        start_date, end_date, classification_results
    )
    
    print("\n" + "="*70)
    print("2️⃣  CREATING HEATMAP VISUALIZATION")
    print("="*70)
    heatmap_output = visualizer.create_comparison_heatmap(
        bin_mask, str(t0_path), str(t1_path),
        start_date, end_date
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\n📂 Generated files:")
    print(f"   1. {main_output} (Main visualization)")
    print(f"   2. {heatmap_output} (Heatmap)")
    print(f"\n🎨 All outputs are publication quality (300 DPI)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
