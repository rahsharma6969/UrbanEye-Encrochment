# ml/scripts/test_credentials.py
"""
Test and debug Sentinel Hub credentials
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sentinelhub import SHConfig

def test_credentials():
    """Test Sentinel Hub credentials"""
    
    print("="*70)
    print("🔐 TESTING SENTINEL HUB CREDENTIALS")
    print("="*70)
    
    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print(f"❌ .env file not found at: {env_path}")
        print(f"\n📝 Create .env file with:")
        print("   SH_CLIENT_ID=your_client_id")
        print("   SH_CLIENT_SECRET=your_client_secret")
        return False
    
    load_dotenv(env_path)
    
    client_id = os.getenv('SH_CLIENT_ID', '').strip()
    client_secret = os.getenv('SH_CLIENT_SECRET', '').strip()
    
    print(f"\n📂 .env file location: {env_path}")
    print(f"🔑 Client ID found: {bool(client_id)} ({'*' * min(len(client_id), 20) if client_id else 'MISSING'})")
    print(f"🔑 Client Secret found: {bool(client_secret)} ({'*' * min(len(client_secret), 20) if client_secret else 'MISSING'})")
    
    if not client_id or not client_secret:
        print("\n❌ Credentials missing!")
        print("\n📝 Get credentials from:")
        print("   https://shapps.dataspace.copernicus.eu/dashboard/#/")
        return False
    
    # Create config
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    
    # Test authentication
    print("\n🔄 Testing authentication...")
    
    try:
        from sentinelhub import SentinelHubSession
        session = SentinelHubSession(config=config)
        token = session.token
        
        print("✅ Authentication successful!")
        print(f"🎫 Access token obtained (length: {len(token)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check credentials at: https://shapps.dataspace.copernicus.eu/")
        print("   2. Ensure OAuth client has proper permissions")
        print("   3. Try regenerating credentials")
        return False


if __name__ == "__main__":
    success = test_credentials()
    
    if success:
        print("\n" + "="*70)
        print("✅ CREDENTIALS VALID - Ready to fetch data!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ FIX CREDENTIALS BEFORE PROCEEDING")
        print("="*70)
