
import os
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np

def get_env_path():
    """Get the path to .env file"""
    possible_paths = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / ".env",
        Path("ml/.env"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def load_credentials():
    """Load credentials from .env file"""
    env_path = get_env_path()
    
    if not env_path:
        raise ValueError(f"❌ .env file not found. Expected at: ml/.env")
    
    load_dotenv(env_path)
    
    client_id = os.getenv('SH_CLIENT_ID', '').strip()
    client_secret = os.getenv('SH_CLIENT_SECRET', '').strip()
    
    return client_id, client_secret

def get_access_token(client_id, client_secret):
    """Get OAuth access token from Copernicus"""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    print("   🔐 Requesting access token...")
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    
    token = response.json()["access_token"]
    print("   ✅ Access token obtained")
    return token

def fetch_sentinel_image(aoi, date, output_path, band_multiplier=2.5):
    """
    Fetch Sentinel-2 image using direct API calls
    
    Args:
        aoi: [lon_min, lat_min, lon_max, lat_max]
        date: Date string in format "YYYY-MM-DD"
        output_path: Path to save the image
        band_multiplier: Brightness multiplier for RGB bands (default: 2.5)
    
    Returns:
        numpy.ndarray: Image array if successful, None otherwise
    """
    try:
        # Load credentials and get token
        client_id, client_secret = load_credentials()
        access_token = get_access_token(client_id, client_secret)
        
        # Validate and parse date
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Invalid date format: {date}. Expected YYYY-MM-DD")
            return None
        
        # Set time range (±7 days for cloud-free imagery)
        time_range_start = (target_date - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
        time_range_end = (target_date + timedelta(days=7)).strftime("%Y-%m-%dT23:59:59Z")
        
        print(f"   🔍 Searching: {time_range_start} to {time_range_end}")
        print("   📡 Requesting data from Copernicus Data Space...")
        
        # Build request payload
        evalscript = f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: ["B02", "B03", "B04"],
                    output: {{ bands: 3 }}
                }};
            }}
            
            function evaluatePixel(sample) {{
                return [{band_multiplier} * sample.B04, {band_multiplier} * sample.B03, {band_multiplier} * sample.B02];
            }}
        """
        
        payload = {
            "input": {
                "bounds": {
                    "bbox": aoi,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                    }
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": time_range_start,
                                "to": time_range_end
                            },
                            "mosaickingOrder": "leastCC"
                        }
                    }
                ]
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/png"
                        }
                    }
                ]
            },
            "evalscript": evalscript
        }
        
        # Make request to Copernicus Data Space
        api_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "image/png"
        }
        
        print("   ⏳ Fetching image data...")
        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            # Convert response to image
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            # Save image
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                img.save(output_path)
                print(f"   💾 Saved to: {output_path}")
            
            print(f"   ✅ Image fetched successfully! Shape: {img_array.shape}")
            return img_array
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            print("\n🔍 Troubleshooting:")
            print("   • Check if date has cloud-free imagery")
            print("   • Verify AOI is within valid bounds")
            print("   • Try a different date range")
            return None
            
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if e.response.status_code == 401:
            print("   🔑 Authentication failed - check your credentials")
        elif e.response.status_code == 403:
            print("   ⚠️  Access denied - check your account permissions")
        elif e.response.status_code == 429:
            print("   ⏳ Rate limit exceeded - wait a moment and try again")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_credentials():
    """Test if credentials are valid"""
    print("\n🔍 Testing Copernicus Data Space Credentials")
    print("="*60)
    
    try:
        client_id, client_secret = load_credentials()
        
        if not client_id or not client_secret:
            print("❌ No credentials found in .env")
            return False
        
        print(f"✓ Client ID: {client_id[:10]}...{client_id[-4:]}")
        print(f"✓ Client Secret: {client_secret[:10]}...***")
        
        # Test authentication
        try:
            token = get_access_token(client_id, client_secret)
            if token:
                print(f"✅ Token obtained (length: {len(token)} chars)")
                return True
            else:
                print("❌ No token received")
                return False
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Sentinel-2 Image Fetch")
    print("🌍 Using Direct Copernicus Data Space API\n")
    
    # Test credentials
    if not test_credentials():
        print("\n" + "="*60)
        print("🔗 Get FREE credentials from:")
        print("   https://shapps.dataspace.copernicus.eu/dashboard/")
        print("="*60)
        exit(1)
    
    # Test fetch
    print("\n" + "="*60)
    print("Testing image fetch...")
    print("="*60)
    
    # Test with Paris (known good location)
    test_aoi = [2.3, 48.85, 2.35, 48.87]
    test_date = "2024-07-15"
    
    result = fetch_sentinel_image(test_aoi, test_date, "test_paris.png")
    
    if result is not None:
        print("\n✅ SUCCESS! Image saved as test_paris.png")
    else:
        print("\n❌ Failed to fetch image")