# scripts/run_tests.py
"""
Run multiple test cases for different area types
"""

import subprocess
import shutil
from pathlib import Path
import time

# Test cases with different urban densities
tests = [
    {
        "name": "Dense_Urban_BKC",
        "coords": [19.05, 72.86, 19.08, 72.88],
        "dates": ["2024-01-15", "2024-07-15"],
        "expected": "MAN-MADE"
    },
    {
        "name": "Suburban_Thane",
        "coords": [19.18, 72.96, 19.21, 72.99],
        "dates": ["2024-02-01", "2024-08-01"],
        "expected": "MIXED"
    },
    {
        "name": "Rural_Area",
        "coords": [19.35, 73.05, 19.38, 73.08],
        "dates": ["2024-01-01", "2024-07-01"],
        "expected": "NATURAL"
    }
]

def main():
    print("="*70)
    print("🧪 RUNNING MULTIPLE TEST CASES")
    print("="*70)
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/3: {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"{'='*70}\n")
        
        coords = test['coords']
        dates = test['dates']
        
        # ✨ FIXED: Correct path resolution
        # Get the root directory (UrbanEye)
        root_dir = Path(__file__).parent.parent
        pipeline_script = root_dir / "scripts" / "run_pipeline.py"
        
        # Build command
        cmd = [
            "python",
            str(pipeline_script),  # Full path to run_pipeline.py
            str(coords[0]), str(coords[1]), str(coords[2]), str(coords[3]),
            dates[0], dates[1]
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        
        # Run pipeline
        result = subprocess.run(cmd, cwd=str(root_dir))
        
        if result.returncode != 0:
            print(f"\n⚠️  Test {i} had errors, but continuing...")
        
        # Create test results directory
        test_dir = root_dir / "presentation" / f"test_{i}_{test['name']}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy results to test directory
        outputs_dir = root_dir / "outputs"
        files_to_copy = [
            "change_map.png",
            "classification_results.json",
            "classification_report.txt",
            "report.txt"
        ]
        
        for filename in files_to_copy:
            src = outputs_dir / filename
            if src.exists():
                dst = test_dir / filename
                shutil.copy(src, dst)
                print(f"  Copied: {filename}")
        
        print(f"\n✅ Test {i} results saved to: {test_dir}")
        
        # Wait a bit between tests
        if i < len(tests):
            print("\n⏳ Waiting 5 seconds before next test...")
            time.sleep(5)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print(f"\n📂 Check results in: presentation/ folder")
    print("\nTest results:")
    for i, test in enumerate(tests, 1):
        test_dir = Path("presentation") / f"test_{i}_{test['name']}"
        print(f"  {i}. {test['name']}: {test_dir}")


if __name__ == "__main__":
    main()
