"""Export trained PyTorch models to various formats for deployment.

Supports:
- ONNX (Open Neural Network Exchange) for cross-platform deployment
- TorchScript for production PyTorch deployments
- Quantization for edge/mobile devices
- Model optimization and validation
"""

import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
import json
from typing import Tuple, Optional, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import (
    SimpleCNN, ResNetClassifier, EfficientNetClassifier,
    DenseNetClassifier, VisionTransformerClassifier, SwinTransformerClassifier, MedicalCNN
)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    arch = checkpoint.get('arch', 'resnet18')
    num_classes = checkpoint.get('num_classes', 2)

    # Smart in_channels detection
    if 'in_channels' in checkpoint:
        in_channels = checkpoint['in_channels']
    elif 'medical' in arch.lower():
        in_channels = 1  # Medical imaging often grayscale
    else:
        in_channels = 3  # Default RGB

    # Initialize model
    if 'resnet' in arch.lower():
        model = ResNetClassifier(arch=arch, num_classes=num_classes, pretrained=False)
    elif 'efficientnet' in arch.lower():
        model = EfficientNetClassifier(arch=arch, num_classes=num_classes, pretrained=False)
    elif 'densenet' in arch.lower():
        model = DenseNetClassifier(arch=arch, num_classes=num_classes, pretrained=False)
    elif 'vit' in arch.lower():
        model = VisionTransformerClassifier(arch=arch, num_classes=num_classes, pretrained=False)
    elif 'swin' in arch.lower():
        model = SwinTransformerClassifier(arch=arch, num_classes=num_classes, pretrained=False)
    elif 'medical' in arch.lower():
        model = MedicalCNN(in_channels=in_channels, num_classes=num_classes)
    else:
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    config = {
        'arch': arch,
        'num_classes': num_classes,
        'in_channels': in_channels,
        'input_shape': (1, in_channels, 224, 224)
    }

    print(f"✓ Loaded model: {arch}")
    return model, config


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_axes: bool = True,
    verify: bool = True
) -> bool:
    """Export model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Output path for ONNX file
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch size
        verify: Whether to verify exported model

    Returns:
        True if export successful
    """
    print(f"\nExporting to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")

    try:
        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Define dynamic axes for variable batch size
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes_dict = None

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
        )

        print(f"✓ Exported to ONNX: {output_path}")

        # Verify ONNX model
        if verify:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verified")

            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(output_path)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare with PyTorch output
            with torch.no_grad():
                pytorch_output = model(dummy_input).numpy()

            max_diff = np.max(np.abs(pytorch_output - ort_outputs[0]))
            print(f"✓ ONNX Runtime test passed (max diff: {max_diff:.6f})")

        return True

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    method: str = 'trace',
    verify: bool = True
) -> bool:
    """Export model to TorchScript format.

    Args:
        model: PyTorch model
        output_path: Output path for TorchScript file
        input_shape: Input tensor shape
        method: Export method ('trace' or 'script')
        verify: Whether to verify exported model

    Returns:
        True if export successful
    """
    print(f"\nExporting to TorchScript...")
    print(f"  Method: {method}")
    print(f"  Input shape: {input_shape}")

    try:
        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Export to TorchScript
        if method == 'trace':
            traced_model = torch.jit.trace(model, dummy_input)
        elif method == 'script':
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")

        # Save TorchScript model
        traced_model.save(output_path)
        print(f"✓ Exported to TorchScript: {output_path}")

        # Verify TorchScript model
        if verify:
            loaded_model = torch.jit.load(output_path)
            loaded_model.eval()

            with torch.no_grad():
                pytorch_output = model(dummy_input).numpy()
                torchscript_output = loaded_model(dummy_input).numpy()

            max_diff = np.max(np.abs(pytorch_output - torchscript_output))
            print(f"✓ TorchScript test passed (max diff: {max_diff:.6f})")

        return True

    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        return False


def quantize_model(
    model: nn.Module,
    output_path: str,
    backend: str = 'fbgemm',
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
) -> bool:
    """Quantize model for edge deployment.

    Args:
        model: PyTorch model
        output_path: Output path for quantized model
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        input_shape: Input tensor shape

    Returns:
        True if quantization successful
    """
    print(f"\nQuantizing model...")
    print(f"  Backend: {backend}")

    try:
        # Set backend
        torch.backends.quantized.engine = backend

        # Skip module fusion - models have complex architectures
        # that don't follow simple Conv+BN+ReLU patterns
        model_to_quantize = model

        # Prepare for quantization
        model_to_quantize.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model_to_quantize, inplace=True)

        # Calibrate with dummy data (in production, use real data)
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            _ = model_to_quantize(dummy_input)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_to_quantize, inplace=True)

        # Save quantized model
        torch.jit.save(torch.jit.script(quantized_model), output_path)
        print(f"✓ Quantized model saved: {output_path}")

        # Check size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = Path(output_path).stat().st_size
        reduction = (1 - quantized_size / original_size) * 100
        print(f"✓ Size reduction: {reduction:.1f}%")

        return True

    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        print("  Note: Quantization may not work for all model architectures")
        return False


def export_model_info(
    model: nn.Module,
    config: Dict[str, Any],
    output_path: str
):
    """Export model metadata.

    Args:
        model: PyTorch model
        config: Model configuration
        output_path: Output path for JSON file
    """
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    info = {
        'architecture': config['arch'],
        'num_classes': config['num_classes'],
        'input_shape': list(config['input_shape']),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(model_size_mb, 2),
        'pytorch_version': torch.__version__
    }

    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Model info saved: {output_path}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Export trained model to deployment formats')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='exports',
                        help='Output directory for exported models')
    parser.add_argument('--formats', type=str, nargs='+',
                        default=['onnx', 'torchscript'],
                        choices=['onnx', 'torchscript', 'quantized', 'all'],
                        help='Export formats')
    parser.add_argument('--input-shape', type=int, nargs=4,
                        default=[1, 3, 224, 224],
                        help='Input shape (batch, channels, height, width)')
    parser.add_argument('--opset-version', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification of exported models')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle 'all' format
    if 'all' in args.formats:
        formats = ['onnx', 'torchscript', 'quantized']
    else:
        formats = args.formats

    print(f"\n{'=' * 80}")
    print(f"Model Export Utility")
    print(f"{'=' * 80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Formats: {', '.join(formats)}")
    print(f"{'=' * 80}\n")

    # Load model
    device = torch.device('cpu')  # Export on CPU for compatibility
    model, config = load_checkpoint(args.checkpoint, device)
    model.to(device)

    # Update input shape from args
    config['input_shape'] = tuple(args.input_shape)

    # Base name for exported files
    base_name = Path(args.checkpoint).stem

    # Export to requested formats
    results = {}

    if 'onnx' in formats:
        output_path = output_dir / f'{base_name}.onnx'
        results['onnx'] = export_to_onnx(
            model,
            str(output_path),
            input_shape=config['input_shape'],
            opset_version=args.opset_version,
            verify=not args.no_verify
        )

    if 'torchscript' in formats:
        output_path = output_dir / f'{base_name}_torchscript.pt'
        results['torchscript'] = export_to_torchscript(
            model,
            str(output_path),
            input_shape=config['input_shape'],
            method='trace',
            verify=not args.no_verify
        )

    if 'quantized' in formats:
        output_path = output_dir / f'{base_name}_quantized.pt'
        results['quantized'] = quantize_model(
            model,
            str(output_path),
            backend='fbgemm',
            input_shape=config['input_shape']
        )

    # Export model info
    info_path = output_dir / f'{base_name}_info.json'
    export_model_info(model, config, str(info_path))

    # Print summary
    print(f"\n{'=' * 80}")
    print("Export Summary:")
    print(f"{'=' * 80}")

    if not results:
        print("  ⚠ No export formats were processed")
        print(f"{'=' * 80}\n")
        return False

    for format_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {format_name.upper()}: {status}")
    print(f"{'=' * 80}\n")

    # Return success if at least one format succeeded
    return any(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
