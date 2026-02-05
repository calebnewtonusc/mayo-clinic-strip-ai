"""Test all new production features.

Tests:
1. Distributed training setup
2. MLflow tracking
3. Model export (ONNX, TorchScript)
4. Deployment scripts existence
5. Configuration files
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import ResNetClassifier, EfficientNetClassifier
from src.models.ensemble import EnsembleVoting, create_ensemble_from_checkpoints
from src.training.advanced_trainer import AdvancedTrainer
from src.training.distributed_trainer import DistributedTrainer, setup_distributed, cleanup_distributed, get_world_info


class TestDistributedTraining:
    """Test distributed training components."""

    def test_world_info_single_process(self):
        """Test world info without distributed setup."""
        info = get_world_info()
        assert info['rank'] == 0
        assert info['world_size'] == 1
        assert info['local_rank'] == 0
        print("✓ World info works in single process mode")

    def test_distributed_trainer_initialization(self):
        """Test DistributedTrainer can be initialized."""
        model = ResNetClassifier(arch='resnet18', num_classes=2, pretrained=False)

        # Create dummy data loaders
        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 2, (100,))
        )
        train_loader = DataLoader(dummy_data, batch_size=16)
        val_loader = DataLoader(dummy_data, batch_size=16)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Initialize trainer (world_size=1 for testing)
        trainer = DistributedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            rank=0,
            world_size=1,
            num_epochs=1
        )

        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert trainer.is_main_process
        print("✓ DistributedTrainer initializes correctly")


class TestMLflowTracking:
    """Test MLflow experiment tracking."""

    def test_mlflow_tracker_import(self):
        """Test MLflowTracker can be imported."""
        from src.training.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            experiment_name="test-experiment",
            run_name="test-run",
            tracking_uri="./test_mlruns"
        )

        assert tracker.experiment_name == "test-experiment"
        assert tracker.run_name == "test-run"
        print("✓ MLflowTracker imports and initializes")

    def test_mlflow_tracker_context(self):
        """Test MLflowTracker context manager."""
        from src.training.mlflow_tracker import MLflowTracker
        import shutil

        tracker = MLflowTracker(
            experiment_name="test-experiment-context",
            run_name="test-run-context",
            tracking_uri="./test_mlruns"
        )

        try:
            with tracker:
                # Log some dummy data
                tracker.log_params({'test_param': 1})
                tracker.log_metric('test_metric', 0.5)
                run_id = tracker.get_run_id()
                assert run_id is not None

            print("✓ MLflowTracker context manager works")
        finally:
            # Cleanup
            if Path('./test_mlruns').exists():
                shutil.rmtree('./test_mlruns')


class TestModelExport:
    """Test model export functionality."""

    def test_export_script_exists(self):
        """Test export script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'export_model.py'
        assert script_path.exists(), "Export script not found"
        print("✓ Export script exists")

    def test_onnx_export_capability(self):
        """Test ONNX export works."""
        import tempfile
        import onnx

        model = ResNetClassifier(arch='resnet18', num_classes=2, pretrained=False)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            temp_path = f.name

        try:
            # Export to ONNX
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model,
                dummy_input,
                temp_path,
                opset_version=14,
                input_names=['input'],
                output_names=['output']
            )

            # Verify ONNX model
            onnx_model = onnx.load(temp_path)
            onnx.checker.check_model(onnx_model)

            print("✓ ONNX export works")
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_torchscript_export_capability(self):
        """Test TorchScript export works."""
        import tempfile

        model = ResNetClassifier(arch='resnet18', num_classes=2, pretrained=False)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            # Export to TorchScript
            dummy_input = torch.randn(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(temp_path)

            # Load and test
            loaded_model = torch.jit.load(temp_path)
            output = loaded_model(dummy_input)
            assert output.shape == (1, 2)

            print("✓ TorchScript export works")
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()


class TestDeploymentInfrastructure:
    """Test deployment scripts and configurations."""

    def test_deployment_scripts_exist(self):
        """Test deployment scripts are present."""
        deploy_dir = Path(__file__).parent.parent / 'deploy'

        required_files = [
            'deploy.sh',
            'shutdown.sh',
            'docker-compose-full.yml',
            'prometheus.yml',
            'api_with_metrics.py'
        ]

        for filename in required_files:
            filepath = deploy_dir / filename
            assert filepath.exists(), f"{filename} not found"

        print("✓ All deployment scripts exist")

    def test_deployment_scripts_executable(self):
        """Test deployment scripts are executable."""
        deploy_dir = Path(__file__).parent.parent / 'deploy'

        scripts = ['deploy.sh', 'shutdown.sh']

        for script in scripts:
            filepath = deploy_dir / script
            assert filepath.exists()
            # Check if executable bit is set
            import os
            assert os.access(filepath, os.X_OK), f"{script} is not executable"

        print("✓ Deployment scripts are executable")

    def test_docker_compose_valid(self):
        """Test docker-compose file is valid YAML."""
        import yaml

        compose_file = Path(__file__).parent.parent / 'deploy' / 'docker-compose-full.yml'

        with open(compose_file, 'r') as f:
            config = yaml.safe_load(f)

        # Check required services
        assert 'services' in config
        assert 'api' in config['services']
        assert 'prometheus' in config['services']
        assert 'grafana' in config['services']

        print("✓ Docker Compose file is valid")

    def test_prometheus_config_valid(self):
        """Test Prometheus config is valid YAML."""
        import yaml

        prom_file = Path(__file__).parent.parent / 'deploy' / 'prometheus.yml'

        with open(prom_file, 'r') as f:
            config = yaml.safe_load(f)

        assert 'scrape_configs' in config
        assert len(config['scrape_configs']) > 0

        print("✓ Prometheus config is valid")

    def test_grafana_provisioning_exists(self):
        """Test Grafana provisioning files exist."""
        deploy_dir = Path(__file__).parent.parent / 'deploy'

        required_files = [
            'grafana-provisioning/datasources/prometheus.yml',
            'grafana-provisioning/dashboards/dashboard.yml',
            'grafana-dashboards/mayo-api-dashboard.json'
        ]

        for filepath in required_files:
            full_path = deploy_dir / filepath
            assert full_path.exists(), f"{filepath} not found"

        print("✓ Grafana provisioning files exist")


class TestCICDConfiguration:
    """Test CI/CD workflow configuration."""

    def test_github_workflow_exists(self):
        """Test GitHub Actions workflow exists."""
        workflow_file = Path(__file__).parent.parent / '.github' / 'workflows' / 'ci-cd.yml'
        assert workflow_file.exists(), "CI/CD workflow not found"
        print("✓ GitHub Actions workflow exists")

    def test_github_workflow_valid(self):
        """Test GitHub Actions workflow is valid YAML."""
        import yaml

        workflow_file = Path(__file__).parent.parent / '.github' / 'workflows' / 'ci-cd.yml'

        with open(workflow_file, 'r') as f:
            config = yaml.safe_load(f)

        assert 'name' in config
        assert 'on' in config or 'true' in config  # 'on' or 'true' for triggers
        assert 'jobs' in config

        # Check for key jobs
        jobs = config['jobs']
        expected_jobs = ['code-quality', 'test', 'security', 'docker']

        for job in expected_jobs:
            assert job in jobs, f"Job '{job}' not found in CI/CD workflow"

        print("✓ GitHub Actions workflow is valid")

    def test_precommit_config_exists(self):
        """Test pre-commit config exists."""
        precommit_file = Path(__file__).parent.parent / '.pre-commit-config.yaml'
        assert precommit_file.exists(), "Pre-commit config not found"
        print("✓ Pre-commit config exists")

    def test_precommit_config_valid(self):
        """Test pre-commit config is valid."""
        import yaml

        precommit_file = Path(__file__).parent.parent / '.pre-commit-config.yaml'

        with open(precommit_file, 'r') as f:
            config = yaml.safe_load(f)

        assert 'repos' in config
        assert len(config['repos']) > 0

        # Check for key hooks
        hooks = []
        for repo in config['repos']:
            if 'hooks' in repo:
                hooks.extend([hook['id'] for hook in repo['hooks']])

        expected_hooks = ['black', 'isort', 'flake8', 'mypy']
        for hook in expected_hooks:
            assert hook in hooks, f"Hook '{hook}' not found in pre-commit config"

        print("✓ Pre-commit config is valid")


class TestMakefile:
    """Test Makefile configuration."""

    def test_makefile_exists(self):
        """Test Makefile exists."""
        makefile = Path(__file__).parent.parent / 'Makefile'
        assert makefile.exists(), "Makefile not found"
        print("✓ Makefile exists")

    def test_makefile_has_key_targets(self):
        """Test Makefile has required targets."""
        makefile = Path(__file__).parent.parent / 'Makefile'

        with open(makefile, 'r') as f:
            content = f.read()

        required_targets = [
            'help', 'install', 'test', 'lint', 'format',
            'train', 'deploy-docker', 'deploy-local', 'export'
        ]

        for target in required_targets:
            assert f'{target}:' in content or f'.PHONY: {target}' in content, \
                f"Target '{target}' not found in Makefile"

        print("✓ Makefile has all required targets")


class TestDocumentation:
    """Test documentation files."""

    def test_production_guide_exists(self):
        """Test PRODUCTION_GUIDE.md exists."""
        guide = Path(__file__).parent.parent / 'PRODUCTION_GUIDE.md'
        assert guide.exists(), "PRODUCTION_GUIDE.md not found"
        print("✓ PRODUCTION_GUIDE.md exists")

    def test_enhancements_doc_exists(self):
        """Test ENHANCEMENTS.md exists."""
        doc = Path(__file__).parent.parent / 'ENHANCEMENTS.md'
        assert doc.exists(), "ENHANCEMENTS.md not found"
        print("✓ ENHANCEMENTS.md exists")

    def test_readme_updated(self):
        """Test README.md exists and has content."""
        readme = Path(__file__).parent.parent / 'README.md'
        assert readme.exists(), "README.md not found"

        with open(readme, 'r') as f:
            content = f.read()

        assert len(content) > 1000, "README.md seems too short"
        print("✓ README.md exists and has content")


def run_all_tests():
    """Run all production feature tests."""
    print("\n" + "="*80)
    print("PRODUCTION FEATURES TEST SUITE")
    print("="*80 + "\n")

    test_classes = [
        TestDistributedTraining,
        TestMLflowTracking,
        TestModelExport,
        TestDeploymentInfrastructure,
        TestCICDConfiguration,
        TestMakefile,
        TestDocumentation
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{'='*80}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*80}")

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                passed += 1
            except Exception as e:
                failed += 1
                error_msg = f"{test_class.__name__}.{method_name}: {str(e)}"
                errors.append(error_msg)
                print(f"✗ {method_name} FAILED: {e}")

    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\nFailed tests:")
        for error in errors:
            print(f"  - {error}")

    print(f"{'='*80}\n")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
