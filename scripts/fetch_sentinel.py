# ml/scripts/fetch_sentinel.py
"""
Fetch Sentinel-2 images - CLOUD-FREE VERSION
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions
)
from dotenv import load_dotenv
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocess.enhance_satellite import enhance_for_change_detection


def init_sentinel_config():
    """Initialize Sentinel Hub configuration"""
    config = SHConfig()
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        raise ValueError(f"❌ .env file not found at {env_path}")
    load_dotenv(env_path)
    
    config.sh_client_id = os.getenv('SH_CLIENT_ID', '').strip()
    config.sh_client_secret = os.getenv('SH_CLIENT_SECRET', '').strip()
    
    if not all([config.sh_client_id, config.sh_client_secret]):
        raise ValueError("❌ Missing Sentinel Hub credentials!")
    
    print("✅ Sentinel Hub config loaded")
    return config


def fetch_sentinel_image(aoi, date, output_path=None, enhance=True):
    """
    Fetch cloud-free Sentinel-2 image
    """
    try:
        config = init_sentinel_config()
        bbox = BBox(bbox=aoi, crs=CRS.WGS84)
        
        # Calculate proper image size
        size = bbox_to_dimensions(bbox, resolution=10)
        print(f"📐 Image size: {size}")
        
        target_date = datetime.strptime(date, "%Y-%m-%d")
        # ✨ Wider time window for better cloud-free chance
        time_interval = (
            (target_date - timedelta(days=45)).strftime("%Y-%m-%d"),
            (target_date + timedelta(days=45)).strftime("%Y-%m-%d")
        )
        
        print(f"🔍 Searching: {time_interval[0]} to {time_interval[1]}")
        
        # ✨ IMPROVED EVALSCRIPT with cloud filtering
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B03", "B02", "SCL", "dataMask"],
            output: { bands: 4 }
          };
        }
        
        function evaluatePixel(sample) {
          // SCL values: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus
          if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) {
            return [0, 0, 0, 0];  // Mask clouds
          }
          
          // Apply gain
          let gain = 2.5;
          return [
            Math.min(1, gain * sample.B04),
            Math.min(1, gain * sample.B03),
            Math.min(1, gain * sample.B02),
            sample.dataMask
          ];
        }
        """
        
        # ✨ Request with strict cloud filtering
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    maxcc=0.2,  # ✨ Only 20% cloud coverage max
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=size,
            config=config
        )
        
        print("📡 Requesting data...")
        
        try:
            data = request.get_data()
        except Exception as e:
            print(f"❌ API Error: {str(e)}")
            return None, None
        
        if not data or len(data) == 0:
            print("❌ No cloud-free data available")
            print("   Try different dates or larger time window")
            return None, None
        
        img = data[0]
        
        # Handle 4-channel image
        if len(img.shape) == 3 and img.shape[2] == 4:
            rgb = img[:, :, :3]
            mask = img[:, :, 3]
            
            valid_pixels_pct = (mask > 0).sum() / mask.size * 100
            print(f"📊 Valid pixels: {valid_pixels_pct:.1f}%")
            
            if valid_pixels_pct < 50:
                print("❌ Less than 50% valid pixels (too much cloud/no-data)")
                return None, None
            
            if rgb.max() <= 1.0:
                img_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            if img.max() <= 1.0:
                img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        
        # Validate
        mean_val = img_uint8.mean()
        var_val = np.var(img_uint8)
        
        print(f"📊 Mean: {mean_val:.2f}, Variance: {var_val:.2f}")
        
        if mean_val < 10 or mean_val > 240:
            print("❌ Invalid brightness")
            return None, None
        
        if var_val < 100:
            print("❌ No variation (likely all cloud or corrupted)")
            return None, None
        
        print("✅ Image validation passed")
        
        # Get actual date
        actual_date = date
        try:
            timestamps = request.get_timestamps()
            if timestamps:
                actual_date = timestamps[0].strftime("%Y-%m-%d")
                print(f"📅 Actual date: {actual_date}")
        except:
            pass
        
        # Enhance
        if enhance:
            print("🎨 Enhancing...")
            try:
                img_uint8 = enhance_for_change_detection(img_uint8)
            except Exception as e:
                print(f"⚠️  Enhancement skipped: {e}")
        
        # Save
        if output_path:
            from PIL import Image as PILImage
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            PILImage.fromarray(img_uint8).save(output_path)
            print(f"💾 Saved: {output_path}")
        
        return img_uint8, actual_date
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
