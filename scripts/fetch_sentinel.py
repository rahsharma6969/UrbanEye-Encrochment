# ml/scripts/fetch_sentinel.py
"""
Fetch Sentinel-2 images from Copernicus Data Space Ecosystem
✅ Updated for new Copernicus authentication
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
    MimeType
)
from dotenv import load_dotenv

def init_sentinel_config():
    """Initialize and validate Sentinel Hub configuration"""
    config = SHConfig()
    
    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        raise ValueError(f"❌ .env file not found at {env_path}")
    
    load_dotenv(env_path)
    
    # Get credentials
    config.sh_client_id = os.getenv('SH_CLIENT_ID', '').strip()
    config.sh_client_secret = os.getenv('SH_CLIENT_SECRET', '').strip()
    config.instance_id = os.getenv('SH_INSTANCE_ID', '').strip()
    
    # Validate credentials
    if not all([config.sh_client_id, config.sh_client_secret, config.instance_id]):
        raise ValueError("❌ Missing required environment variables in .env file")
    
    return config

def fetch_sentinel_image(aoi, date, output_path):
    """Fetch Sentinel-2 image for given AOI and date"""
    try:
        # Initialize configuration
        config = init_sentinel_config()
        
        # Initial time range
        target_date = datetime.strptime(date, "%Y-%m-%d")
        time_range = (
            target_date - timedelta(days=3),
            target_date + timedelta(days=3)
        )
        
        print(f"   🔍 Searching: {time_range[0].strftime('%Y-%m-%d')} to {time_range[1].strftime('%Y-%m-%d')}")
        print("   📡 Requesting data from Copernicus...")
        
        # Create bounding box
        bbox = BBox(bbox=aoi, crs=CRS.WGS84)
        
        def create_request(time_interval):
            return SentinelHubRequest(
                evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["B02", "B03", "B04"],
                            output: { 
                                bands: 3,
                                sampleType: "UINT8"
                            }
                        };
                    }
                    
                    function evaluatePixel(sample) {
                        let gain = 3.5;
                        let rgb = [sample.B04, sample.B03, sample.B02];
                        return rgb.map(v => Math.min(255, v * gain * 255));
                    }
                """,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.PNG)
                ],
                bbox=bbox,
                size=(1024, 1024),
                config=config
            )
        
        # Try with initial time range
        request = create_request(time_range)
        img_list = request.get_data()
        
        # If no valid image, try wider range
        if not img_list or (len(img_list) > 0 and img_list[0].mean() < 1):
            print("⚠️ No valid image found for the specified date, trying wider range...")
            wider_range = (
                target_date - timedelta(days=15),
                target_date + timedelta(days=15)
            )
            request = create_request(wider_range)
            img_list = request.get_data()
        
        if not img_list:
            print("❌ No images found")
            return None
        
        img = img_list[0]
        if img.mean() < 1:
            print("❌ Retrieved image is too dark or invalid")
            return None
        
        # Save image
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            from PIL import Image
            Image.fromarray(img.astype('uint8')).save(output_path)
        
        return img
        
    except Exception as e:
        print(f"\n❌ Error fetching image: {str(e)}")
        return None


def test_credentials():
    """Test if credentials are valid"""
    print("\n🔍 Testing Copernicus Credentials")
    print("="*60)
    
    if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
        print("❌ No credentials found in .env")
        print("\n📝 Create ml/.env file with:")
        print("   SH_CLIENT_ID=your_client_id")
        print("   SH_CLIENT_SECRET=your_client_secret")
        return False
    
    print(f"✓ Client ID: {SH_CLIENT_ID[:10]}...{SH_CLIENT_ID[-4:]}")
    print(f"✓ Client Secret: {SH_CLIENT_SECRET[:10]}...***")
    
    # Test authentication
    try:
        from sentinelhub import SentinelHubSession
        config = get_copernicus_config()
        session = SentinelHubSession(config=config)
        token = session.token
        
        if token:
            print("✅ Authentication successful!")
            print(f"✅ Token obtained (valid for ~1 hour)")
            return True
        else:
            print("❌ Authentication failed - no token")
            return False
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Sentinel Image Fetch\n")
    
    # First test credentials
    if not test_credentials():
        print("\n" + "="*60)
        print("🔗 Get credentials from:")
        print("   https://dataspace.copernicus.eu/")
        print("   → Login/Register")
        print("   → Go to: https://shapps.dataspace.copernicus.eu/dashboard/")
        print("   → Settings → OAuth clients → Create new")
        print("="*60)
        exit(1)
    
    # Test fetch
    print("\n" + "="*60)
    print("Testing image fetch...")
    print("="*60)
    
    test_aoi = [2.3, 48.85, 2.35, 48.87]  # Paris
    test_date = "2023-07-15"
    
    result = fetch_sentinel_image(test_aoi, test_date, "test_paris.png")
    
    if result is not None:
        print("\n✅ SUCCESS! Check test_paris.png")
    else:
        print("\n❌ Failed to fetch image")