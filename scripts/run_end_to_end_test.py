"""End-to-end test of the entire pipeline with dummy data."""

import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        print(f"Command failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ SUCCESS: {description}")
        return True


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          Mayo Clinic STRIP AI - End-to-End Test              ║
║                                                               ║
║  This script will test the entire pipeline with dummy data   ║
╚═══════════════════════════════════════════════════════════════╝
""")

    # Check if we're in the right directory
    if not Path('src').exists() or not Path('scripts').exists():
        print("❌ Error: Must run from project root directory")
        print("cd to mayo-clinic-strip-ai/ first")
        sys.exit(1)

    # Create test data directory
    test_data_dir = Path('data/test_dummy')
    if test_data_dir.exists():
        print(f"🗑️  Removing existing test data directory: {test_data_dir}")
        shutil.rmtree(test_data_dir)

    steps = [
        {
            'cmd': f'python scripts/generate_dummy_data.py --output_dir {test_data_dir}/raw --num_patients 10',
            'desc': 'Step 1: Generate dummy medical imaging data'
        },
        {
            'cmd': f'python scripts/validate_data.py --data_dir {test_data_dir}/raw',
            'desc': 'Step 2: Validate data structure and quality'
        },
        {
            'cmd': f'python scripts/explore_data.py --data_dir {test_data_dir}/raw --output_dir {test_data_dir}/exploration',
            'desc': 'Step 3: Explore data and create visualizations'
        },
        {
            'cmd': f'python scripts/preprocess_data.py --input_dir {test_data_dir}/raw --output_dir {test_data_dir}/processed --target_size 224 224',
            'desc': 'Step 4: Preprocess images'
        },
        {
            'cmd': f'python scripts/create_splits.py --data_dir {test_data_dir}/processed --output_dir {test_data_dir}/splits --seed 42',
            'desc': 'Step 5: Create train/val/test splits'
        },
        {
            'cmd': f'python scripts/test_dataloader.py --data_dir {test_data_dir}/processed --split_file {test_data_dir}/splits/train.json --batch_size 8',
            'desc': 'Step 6: Test DataLoader functionality'
        }
    ]

    failed_steps = []

    for i, step in enumerate(steps, 1):
        success = run_command(step['cmd'], step['desc'])
        if not success:
            failed_steps.append(i)
            print(f"\n⚠️  Step {i} failed, but continuing...")

    # Summary
    print(f"\n\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    if not failed_steps:
        print("✅ ALL TESTS PASSED!")
        print("\nThe entire pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Replace dummy data with real medical images in data/raw/")
        print("2. Run preprocessing and splitting on real data")
        print("3. Start training: python train.py")
    else:
        print(f"❌ {len(failed_steps)} step(s) failed: {failed_steps}")
        print("\nPlease check the error messages above.")

    print(f"\n📁 Test data location: {test_data_dir}")
    print("💡 You can safely delete this directory when done testing")

    return 0 if not failed_steps else 1


if __name__ == '__main__':
    sys.exit(main())
