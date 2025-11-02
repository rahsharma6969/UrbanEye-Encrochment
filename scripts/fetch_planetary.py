# scripts/fetch_planetary.py
"""
Fetch Sentinel-2 from Microsoft Planetary Computer
✅ FREE - No authentication needed
✅ Works immediately
"""

import planetary_computer as pc
import pystac_client
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import rasterio
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

try:
    from PIL import Image as PILImage
except ImportError:
    import PIL.Image as PILImage


def fetch_planetary_image(aoi, date, output_path=None):
    """
    Fetch Sentinel-2 from Planetary Computer
    
    Args:
        aoi: [lon_min, lat_min, lon_max, lat_max]
        date: "YYYY-MM-DD"
        output_path: Where to save
    
    Returns:
        (image_array, actual_date) or (None, None)
    """
    
    print(f"\n{'='*60}")
    print(f"🌍 Fetching from Microsoft Planetary Computer")
    print(f"{'='*60}")
    print(f"📍 AOI: {aoi}")
    print(f"📅 Target date: {date}")
    
    try:
        # Open catalog (no auth needed!)
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        
        # Parse date
        target_date = datetime.strptime(date, "%Y-%m-%d")
        
        # Search with time window
        date_range = f"{(target_date - timedelta(days=15)).strftime('%Y-%m-%d')}/{(target_date + timedelta(days=15)).strftime('%Y-%m-%d')}"
        
        print(f"🔍 Searching: {date_range}")
        
        # Search for images
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=aoi,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 30}}
        )
        
        items = list(search.items())
        
        if not items:
            print("❌ No images found")
            return None, None
        
        print(f"✅ Found {len(items)} images")
        
        # Get first (most recent) item
        item = sorted(items, key=lambda x: x.datetime, reverse=True)[0]
        actual_date = item.datetime.strftime("%Y-%m-%d")
        
        print(f"📅 Using image from: {actual_date}")
        print(f"☁️  Cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')}%")
        
        # Get visual asset (RGB)
        if 'visual' in item.assets:
            asset = item.assets['visual']
        elif 'rendered_preview' in item.assets:
            asset = item.assets['rendered_preview']
        else:
            print("⚠️  Using individual bands...")
            # Fallback: read RGB bands separately
            return fetch_rgb_bands(item, output_path, actual_date, aoi)
        
        # Read the visual asset
        print("📦 Downloading image...")
        
        with rasterio.open(pc.sign(asset.href)) as src:
            # Read and resample to 512x512
            data = src.read(
                out_shape=(src.count, 512, 512),
                resampling=Resampling.bilinear
            )
            
            # Convert to uint8 RGB
            if data.shape[0] == 3:
                img = np.transpose(data, (1, 2, 0))  # CHW -> HWC
            else:
                img = data[0]
            
            # Normalize
            if img.max() > 255:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        print(f"✅ Image downloaded: {img.shape}")
        print(f"📊 Mean: {img.mean():.1f}, Range: [{img.min()}, {img.max()}]")
        
        # Save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(img).save(output_path)
            print(f"💾 Saved: {output_path}")
        
        return img, actual_date
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def fetch_rgb_bands(item, output_path, actual_date, aoi):
    """Fallback: Fetch RGB bands separately"""
    
    print("📦 Fetching RGB bands...")
    
    try:
        import rioxarray
        
        # Get RGB bands
        bands = []
        for band_name in ['B04', 'B03', 'B02']:  # Red, Green, Blue
            if band_name in item.assets:
                with rioxarray.open_rasterio(pc.sign(item.assets[band_name].href)) as band_data:
                    # Clip to AOI
                    band_arr = band_data.rio.clip_box(*aoi).values[0]
                    bands.append(band_arr)
        
        if len(bands) != 3:
            print(f"❌ Could not get all RGB bands (got {len(bands)})")
            return None, None
        
        # Stack bands
        img = np.stack(bands, axis=-1)
        
        # Resize to 512x512
        import cv2
        img = cv2.resize(img, (512, 512))
        
        # Normalize
        img = (img / img.max() * 255).astype(np.uint8)
        
        print(f"✅ RGB bands combined: {img.shape}")
        
        # Save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(img).save(output_path)
            print(f"💾 Saved: {output_path}")
        
        return img, actual_date
        
    except Exception as e:
        print(f"❌ RGB bands fetch failed: {e}")
        return None, None


if __name__ == "__main__":
    print("🧪 Testing Planetary Computer fetch...")
    
    # Mumbai test
    test_aoi = [72.86, 19.05, 72.88, 19.08]
    test_date = "2024-03-01"
    
    img, date = fetch_planetary_image(test_aoi, test_date, "test_pc.png")
    
    if img is not None:
        print(f"\n✅ SUCCESS!")
        print(f"Shape: {img.shape}")
        print(f"Date: {date}")
    else:
        print(f"\n❌ FAILED")
