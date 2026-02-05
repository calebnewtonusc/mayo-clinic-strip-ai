"""Run all tests for the project."""

import subprocess
import sys
from pathlib import Path


def run_tests(test_dir='tests', verbose=True):
    """Run pytest on all tests.

    Args:
        test_dir: Directory containing tests
        verbose: Whether to use verbose output
    """
    test_path = Path(test_dir)

    if not test_path.exists():
        print(f"Error: Test directory {test_dir} not found")
        return 1

    # Build pytest command
    cmd = ['pytest', str(test_path)]

    if verbose:
        cmd.append('-v')

    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(['--cov=src', '--cov-report=term-missing'])
    except ImportError:
        print("Note: pytest-cov not installed. Install for coverage reports.")

    print("Running tests...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run tests
    result = subprocess.run(cmd)

    return result.returncode


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run project tests')
    parser.add_argument('--test-dir', default='tests', help='Test directory')
    parser.add_argument('--quiet', action='store_true', help='Quiet output')
    args = parser.parse_args()

    exit_code = run_tests(
        test_dir=args.test_dir,
        verbose=not args.quiet
    )

    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
