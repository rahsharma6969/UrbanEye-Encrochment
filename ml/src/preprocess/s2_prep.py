import numpy as np
import rioxarray as rxr

GOOD_SCL = {4, 5, 6, 7, 11}  # keep vegetation/bare/water/shadow-ish; exclude clouds 8/9/10

def cloud_mask_from_scl(scl_da):
    """
    Create cloud mask from Sentinel-2 SCL (Scene Classification Layer)
    Returns True for pixels to keep, False for pixels to mask
    """
    # Handle both xarray DataArray and numpy array inputs
    if hasattr(scl_da, 'values'):
        data = scl_da.values
    elif hasattr(scl_da, 'data'):
        data = scl_da.data
    else:
        data = scl_da
    
    # Ensure we have a numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Create mask - True for good pixels to keep
    keep = np.isin(data, list(GOOD_SCL))
    
    # Handle NaN values - keep them as they are (don't mask them further)
    if np.issubdtype(data.dtype, np.floating):
        nan_mask = np.isnan(data)
        keep = keep | nan_mask  # Keep NaN pixels as they are
    
    print(f"  📊 SCL mask stats: {keep.mean():.2%} pixels kept, shape: {keep.shape}")
    return keep

def apply_mask(arr, mask):
    """
    Apply cloud mask to multi-band array
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Multi-band array with shape [C, H, W] where C=channels, H=height, W=width
    mask : numpy.ndarray  
        2D boolean mask with shape [H, W]. True=keep pixel, False=mask pixel
        
    Returns:
    --------
    numpy.ndarray : Masked array with same shape as input
    """
    
    # Validate inputs
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr)}")
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy mask, got {type(mask)}")
    
    # Handle different array dimensions
    if arr.ndim == 2:
        # Single band case [H, W]
        if arr.shape != mask.shape:
            raise ValueError(f"Shape mismatch: arr {arr.shape} vs mask {mask.shape}")
        masked = arr.copy()
        masked[~mask] = np.nan
        return masked
        
    elif arr.ndim == 3:
        # Multi-band case [C, H, W]
        expected_mask_shape = arr.shape[1:3]  # [H, W]
        if mask.shape != expected_mask_shape:
            raise ValueError(f"Shape mismatch: expected mask shape {expected_mask_shape}, got {mask.shape}")
        
        # Apply mask efficiently using broadcasting
        masked = arr.copy()
        masked[:, ~mask] = np.nan  # Broadcasting: [C, H, W][:, ~[H, W]] = [C, H, W]
        return masked
        
    else:
        raise ValueError(f"Unsupported array dimensions: {arr.shape}. Expected 2D or 3D array.")

def validate_arrays_for_masking(arr, mask, pair_id=None):
    """
    Validate that arrays and masks are compatible before masking
    """
    pair_info = f"pair {pair_id}" if pair_id is not None else "array"
    
    try:
        print(f"  🔍 Validating {pair_info}: arr shape {arr.shape}, mask shape {mask.shape}")
        
        if arr.ndim == 3:
            expected_mask_shape = arr.shape[1:3]
        elif arr.ndim == 2:
            expected_mask_shape = arr.shape
        else:
            raise ValueError(f"Unsupported array dimensions: {arr.shape}")
            
        if mask.shape != expected_mask_shape:
            print(f"  ⚠️  Shape mismatch for {pair_info}:")
            print(f"      Array shape: {arr.shape}")  
            print(f"      Mask shape: {mask.shape}")
            print(f"      Expected mask shape: {expected_mask_shape}")
            return False
            
        print(f"  ✅ {pair_info} validation passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation failed for {pair_info}: {e}")
        return False

# Alternative safer masking function that handles shape mismatches
def apply_mask_safe(arr, mask, pair_id=None):
    """
    Safely apply mask with automatic shape handling
    """
    try:
        # Validate first
        if not validate_arrays_for_masking(arr, mask, pair_id):
            # Try to fix common shape issues
            if arr.ndim == 3 and mask.ndim == 2:
                # Standard case - should work
                expected_shape = arr.shape[1:3]
                if mask.shape != expected_shape:
                    # Crop mask to match array
                    h, w = expected_shape
                    mask = mask[:h, :w]
                    print(f"  🔧 Cropped mask to {mask.shape} for {pair_id or 'array'}")
        
        return apply_mask(arr, mask)
        
    except Exception as e:
        print(f"  ❌ Masking failed for {pair_id or 'array'}: {e}")
        # Return unmasked array as fallback
        return arr.copy()

# Helper function to check SCL data quality
def check_scl_quality(scl_da, pair_id=None):
    """Check SCL data for common issues"""
    pair_info = f" for pair {pair_id}" if pair_id is not None else ""
    
    try:
        data = scl_da.values if hasattr(scl_da, 'values') else scl_da
        
        unique_vals = np.unique(data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data)
        print(f"  📊 SCL unique values{pair_info}: {sorted(unique_vals)}")
        
        # Check if we have reasonable SCL values
        valid_scl_range = set(range(0, 12))  # Standard SCL values 0-11
        unexpected = set(unique_vals) - valid_scl_range
        if unexpected:
            print(f"  ⚠️  Unexpected SCL values{pair_info}: {unexpected}")
            
        # Check coverage
        good_pixels = np.isin(data, list(GOOD_SCL))
        coverage = good_pixels.mean() if good_pixels.size > 0 else 0
        print(f"  📊 Good pixel coverage{pair_info}: {coverage:.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SCL quality check failed{pair_info}: {e}")
        return False