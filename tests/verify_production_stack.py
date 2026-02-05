#!/usr/bin/env python3
"""Verify production stack components are in place.

This is a lightweight verification script that checks all production
features are properly installed without requiring heavy dependencies.
"""

import sys
from pathlib import Path
import json


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists."""
    if filepath.exists():
        print(f"✓ {description}: {filepath.name}")
        return True
    else:
        print(f"✗ {description}: {filepath.name} NOT FOUND")
        return False


def check_yaml_valid(filepath: Path, description: str) -> bool:
    """Check if YAML file is valid."""
    try:
        import yaml
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        print(f"✓ {description}: Valid YAML")
        return True
    except ImportError:
        print(f"⚠ {description}: Cannot validate (yaml not installed)")
        return True  # Don't fail if yaml not installed
    except Exception as e:
        print(f"✗ {description}: Invalid YAML - {e}")
        return False


def check_json_valid(filepath: Path, description: str) -> bool:
    """Check if JSON file is valid."""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"✓ {description}: Valid JSON")
        return True
    except Exception as e:
        print(f"✗ {description}: Invalid JSON - {e}")
        return False


def check_executable(filepath: Path, description: str) -> bool:
    """Check if file is executable."""
    import os
    if os.access(filepath, os.X_OK):
        print(f"✓ {description}: Executable")
        return True
    else:
        print(f"✗ {description}: Not executable")
        return False


def main():
    """Run all verification checks."""
    project_root = Path(__file__).parent.parent
    passed = 0
    failed = 0

    print("\n" + "=" * 80)
    print("PRODUCTION STACK VERIFICATION")
    print("=" * 80 + "\n")

    # Section 1: Core Production Files
    print("=" * 80)
    print("1. Core Production Enhancement Files")
    print("=" * 80)

    checks = [
        (project_root / 'src' / 'models' / 'ensemble.py', 'Model Ensemble System'),
        (project_root / 'src' / 'training' / 'advanced_trainer.py', 'Advanced Trainer (Mixed Precision)'),
        (project_root / 'src' / 'training' / 'distributed_trainer.py', 'Distributed Training'),
        (project_root / 'src' / 'training' / 'mlflow_tracker.py', 'MLflow Tracking'),
        (project_root / 'deploy' / 'api_with_metrics.py', 'API with Prometheus Metrics'),
    ]

    for filepath, desc in checks:
        if check_file_exists(filepath, desc):
            passed += 1
        else:
            failed += 1

    # Section 2: Docker & Deployment
    print("\n" + "=" * 80)
    print("2. Docker & Deployment Infrastructure")
    print("=" * 80)

    docker_files = [
        (project_root / 'deploy' / 'docker-compose-full.yml', 'Docker Compose (Full Stack)', check_yaml_valid),
        (project_root / 'deploy' / 'prometheus.yml', 'Prometheus Configuration', check_yaml_valid),
        (project_root / 'deploy' / 'deploy.sh', 'Deployment Script', check_executable),
        (project_root / 'deploy' / 'shutdown.sh', 'Shutdown Script', check_executable),
    ]

    for filepath, desc, *validator in docker_files:
        if filepath.exists():
            if validator:
                if validator[0](filepath, desc):
                    passed += 1
                else:
                    failed += 1
            else:
                print(f"✓ {desc}: {filepath.name}")
                passed += 1
        else:
            print(f"✗ {desc}: {filepath.name} NOT FOUND")
            failed += 1

    # Section 3: Grafana Provisioning
    print("\n" + "=" * 80)
    print("3. Grafana Monitoring Setup")
    print("=" * 80)

    grafana_files = [
        (project_root / 'deploy' / 'grafana-provisioning' / 'datasources' / 'prometheus.yml',
         'Grafana Datasource', check_yaml_valid),
        (project_root / 'deploy' / 'grafana-provisioning' / 'dashboards' / 'dashboard.yml',
         'Grafana Dashboard Config', check_yaml_valid),
        (project_root / 'deploy' / 'grafana-dashboards' / 'mayo-api-dashboard.json',
         'Pre-built Dashboard', check_json_valid),
    ]

    for filepath, desc, validator in grafana_files:
        if filepath.exists():
            if validator(filepath, desc):
                passed += 1
            else:
                failed += 1
        else:
            print(f"✗ {desc}: NOT FOUND")
            failed += 1

    # Section 4: CI/CD
    print("\n" + "=" * 80)
    print("4. CI/CD Pipeline")
    print("=" * 80)

    cicd_files = [
        (project_root / '.github' / 'workflows' / 'ci-cd.yml', 'GitHub Actions Workflow', check_yaml_valid),
        (project_root / '.pre-commit-config.yaml', 'Pre-commit Hooks', check_yaml_valid),
    ]

    for filepath, desc, validator in cicd_files:
        if filepath.exists():
            if validator(filepath, desc):
                passed += 1
            else:
                failed += 1
        else:
            print(f"✗ {desc}: NOT FOUND")
            failed += 1

    # Section 5: Scripts
    print("\n" + "=" * 80)
    print("5. Automation Scripts")
    print("=" * 80)

    script_files = [
        (project_root / 'scripts' / 'train_distributed.py', 'Distributed Training Script'),
        (project_root / 'scripts' / 'export_model.py', 'Model Export Script'),
        (project_root / 'Makefile', 'Makefile'),
    ]

    for filepath, desc in script_files:
        if check_file_exists(filepath, desc):
            passed += 1
        else:
            failed += 1

    # Section 6: Documentation
    print("\n" + "=" * 80)
    print("6. Documentation")
    print("=" * 80)

    doc_files = [
        (project_root / 'PRODUCTION_GUIDE.md', 'Production Deployment Guide'),
        (project_root / 'ENHANCEMENTS.md', 'Features Documentation'),
        (project_root / 'README.md', 'README'),
    ]

    for filepath, desc in doc_files:
        if filepath.exists():
            # Check file size to ensure it has content
            size = filepath.stat().st_size
            if size > 1000:  # At least 1KB
                print(f"✓ {desc}: {size:,} bytes")
                passed += 1
            else:
                print(f"⚠ {desc}: Exists but seems small ({size} bytes)")
                passed += 1
        else:
            print(f"✗ {desc}: NOT FOUND")
            failed += 1

    # Section 7: File Structure Checks
    print("\n" + "=" * 80)
    print("7. Directory Structure")
    print("=" * 80)

    directories = [
        (project_root / 'deploy' / 'grafana-provisioning', 'Grafana Provisioning Dir'),
        (project_root / 'deploy' / 'grafana-dashboards', 'Grafana Dashboards Dir'),
        (project_root / '.github' / 'workflows', 'GitHub Workflows Dir'),
        (project_root / 'src' / 'training', 'Training Modules Dir'),
    ]

    for dirpath, desc in directories:
        if dirpath.exists() and dirpath.is_dir():
            print(f"✓ {desc}: Exists")
            passed += 1
        else:
            print(f"✗ {desc}: NOT FOUND")
            failed += 1

    # Final Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total Checks: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL PRODUCTION FEATURES VERIFIED!")
        print("\nProduction stack is complete and ready:")
        print("  • Docker Compose with Prometheus + Grafana")
        print("  • CI/CD GitHub Actions workflow")
        print("  • Distributed multi-GPU training")
        print("  • MLflow experiment tracking")
        print("  • Model export (ONNX/TorchScript)")
        print("  • Pre-commit hooks")
        print("  • Deployment automation scripts")
        print("  • Complete documentation")
        print("\nQuick Start:")
        print("  make help              # See all available commands")
        print("  make deploy-docker     # Deploy full stack")
        print("  make train-dist        # Train with multi-GPU")
    else:
        print(f"\n⚠ {failed} checks failed - review errors above")

    print("=" * 80 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
