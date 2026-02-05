"""Optimize trained models for deployment (quantization, pruning)."""

import torch
import torch.nn as nn
import torch.quantization as quant
from pathlib import Path
import argparse
import sys
sys.path.append('..')

from src.models.cnn import SimpleCNN, ResNetClassifier, EfficientNetClassifier
from src.utils.helpers import load_config, get_device, count_parameters


def apply_dynamic_quantization(model):
    """Apply dynamic quantization to model.

    Dynamic quantization converts weights to int8 and activations are
    quantized on-the-fly during inference.

    Args:
        model: Model to quantize

    Returns:
        Quantized model
    """
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model


def apply_static_quantization(model, calibration_loader, device):
    """Apply static quantization to model.

    Static quantization requires calibration data to determine optimal
    quantization parameters for both weights and activations.

    Args:
        model: Model to quantize
        calibration_loader: DataLoader with calibration data
        device: Device to run on

    Returns:
        Quantized model
    """
    # Prepare model for quantization
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    model_prepared = quant.prepare(model, inplace=False)

    # Calibrate with sample data
    print("Calibrating model...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 100:  # Use first 100 batches
                break
            images = images.to(device)
            model_prepared(images)

    # Convert to quantized model
    model_quantized = quant.convert(model_prepared, inplace=False)

    return model_quantized


def prune_model(model, amount=0.3):
    """Apply magnitude-based pruning to model.

    Args:
        model: Model to prune
        amount: Fraction of weights to prune (0-1)

    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune

    # Apply structured pruning to convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)

    # Apply pruning to linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

    return model


def remove_pruning_reparametrization(model):
    """Make pruning permanent by removing reparametrization.

    Args:
        model: Pruned model

    Returns:
        Model with pruning made permanent
    """
    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except:
                pass

    return model


def measure_model_size(model, filepath=None):
    """Measure model size in MB.

    Args:
        model: Model to measure
        filepath: Optional path where model is saved

    Returns:
        Model size in MB
    """
    if filepath and Path(filepath).exists():
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    else:
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.tell() / (1024 * 1024)

    return size_mb


def measure_inference_time(model, input_shape=(1, 3, 224, 224), device='cpu', num_runs=100):
    """Measure average inference time.

    Args:
        model: Model to measure
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of inference runs

    Returns:
        Average inference time in milliseconds
    """
    import time

    model.eval()
    model = model.to(device)

    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Measure
    times = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

    return sum(times) / len(times)


def export_to_onnx(model, output_path, input_shape=(1, 3, 224, 224)):
    """Export model to ONNX format.

    Args:
        model: Model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape
    """
    model.eval()
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize model for deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='models/optimized',
                        help='Directory to save optimized models')
    parser.add_argument('--method', type=str, default='all',
                        choices=['quantize', 'prune', 'all'],
                        help='Optimization method')
    parser.add_argument('--prune-amount', type=float, default=0.3,
                        help='Fraction of weights to prune (0-1)')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export to ONNX format')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Initialize model
    model_arch = checkpoint.get('arch', 'resnet18')
    num_classes = checkpoint.get('num_classes', 2)

    if 'resnet' in model_arch:
        model = ResNetClassifier(
            arch=model_arch,
            num_classes=num_classes,
            pretrained=False
        )
    elif 'efficientnet' in model_arch:
        model = EfficientNetClassifier(
            arch=model_arch,
            num_classes=num_classes,
            pretrained=False
        )
    else:
        model = SimpleCNN(in_channels=3, num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nOriginal model: {model_arch}")
    print(f"Parameters: {count_parameters(model):,}")

    # Measure original model
    original_size = measure_model_size(model)
    original_time = measure_inference_time(model, device=str(device))

    print(f"Model size: {original_size:.2f} MB")
    print(f"Inference time: {original_time:.2f} ms")

    results = {
        'original': {
            'size_mb': original_size,
            'inference_time_ms': original_time,
            'parameters': count_parameters(model)
        }
    }

    # Apply optimizations
    if args.method in ['quantize', 'all']:
        print("\n" + "="*60)
        print("APPLYING DYNAMIC QUANTIZATION")
        print("="*60)

        quantized_model = apply_dynamic_quantization(model)

        # Save quantized model
        quantized_path = output_dir / f'{model_arch}_quantized.pth'
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'arch': model_arch,
            'num_classes': num_classes,
            'optimization': 'dynamic_quantization'
        }, quantized_path)

        # Measure quantized model
        quant_size = measure_model_size(quantized_model, quantized_path)
        quant_time = measure_inference_time(quantized_model, device='cpu')  # Quantized models run on CPU

        print(f"Quantized model size: {quant_size:.2f} MB")
        print(f"Quantized inference time: {quant_time:.2f} ms")
        print(f"Size reduction: {(1 - quant_size/original_size)*100:.1f}%")
        print(f"Speedup: {original_time/quant_time:.2f}x")

        results['quantized'] = {
            'size_mb': quant_size,
            'inference_time_ms': quant_time,
            'size_reduction_pct': (1 - quant_size/original_size)*100,
            'speedup': original_time/quant_time
        }

        print(f"\nSaved: {quantized_path}")

    if args.method in ['prune', 'all']:
        print("\n" + "="*60)
        print("APPLYING PRUNING")
        print("="*60)

        pruned_model = prune_model(model, amount=args.prune_amount)

        # Count remaining parameters
        total_params = 0
        nonzero_params = 0
        for param in pruned_model.parameters():
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()

        sparsity = 1 - (nonzero_params / total_params)
        print(f"Sparsity: {sparsity*100:.1f}%")
        print(f"Remaining parameters: {nonzero_params:,} / {total_params:,}")

        # Make pruning permanent
        pruned_model = remove_pruning_reparametrization(pruned_model)

        # Save pruned model
        pruned_path = output_dir / f'{model_arch}_pruned_{int(args.prune_amount*100)}pct.pth'
        torch.save({
            'model_state_dict': pruned_model.state_dict(),
            'arch': model_arch,
            'num_classes': num_classes,
            'optimization': 'pruning',
            'prune_amount': args.prune_amount,
            'sparsity': sparsity
        }, pruned_path)

        # Measure pruned model
        pruned_size = measure_model_size(pruned_model, pruned_path)
        pruned_time = measure_inference_time(pruned_model, device=str(device))

        print(f"Pruned model size: {pruned_size:.2f} MB")
        print(f"Pruned inference time: {pruned_time:.2f} ms")

        results['pruned'] = {
            'size_mb': pruned_size,
            'inference_time_ms': pruned_time,
            'sparsity_pct': sparsity*100,
            'nonzero_params': nonzero_params
        }

        print(f"\nSaved: {pruned_path}")

    # Export to ONNX
    if args.export_onnx:
        print("\n" + "="*60)
        print("EXPORTING TO ONNX")
        print("="*60)

        onnx_path = output_dir / f'{model_arch}.onnx'
        export_to_onnx(model, onnx_path)

        onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"ONNX model size: {onnx_size:.2f} MB")

        results['onnx'] = {
            'size_mb': onnx_size,
            'path': str(onnx_path)
        }

    # Save results
    import json
    results_path = output_dir / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
