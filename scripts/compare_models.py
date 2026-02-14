"""Compare multiple trained models."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_training_history(checkpoint_dir):
    """Load training history from checkpoint directory."""
    results_path = Path(checkpoint_dir).parent / 'results' / 'training_history.json'

    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        history = json.load(f)

    return history


def load_evaluation_results(results_path):
    """Load evaluation results from JSON."""
    if not Path(results_path).exists():
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def plot_training_curves(experiments, save_path=None):
    """Plot training curves for multiple experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name, history in experiments.items():
        if history is None:
            continue

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        axes[0].plot(epochs, history['train_loss'], label=f'{exp_name} (train)', alpha=0.7)
        axes[0].plot(epochs, history['val_loss'], label=f'{exp_name} (val)', alpha=0.7, linestyle='--')

        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], label=f'{exp_name} (train)', alpha=0.7)
        axes[1].plot(epochs, history['val_acc'], label=f'{exp_name} (val)', alpha=0.7, linestyle='--')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_metrics(evaluation_results, save_path=None):
    """Compare evaluation metrics across models."""
    # Extract metrics
    data = []
    for exp_name, results in evaluation_results.items():
        if results is None:
            continue

        metrics = results['metrics']
        for metric_name, value in metrics.items():
            data.append({
                'Model': exp_name,
                'Metric': metric_name,
                'Value': value
            })

    df = pd.DataFrame(data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for grouped bar chart
    pivot_df = df.pivot(index='Metric', columns='Model', values='Value')

    pivot_df.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison - Evaluation Metrics')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print table
    print("\nMetrics Comparison Table:")
    print("="*60)
    print(pivot_df.to_string())
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                        help='Experiment names or checkpoint directories')
    parser.add_argument('--eval_results', type=str, nargs='+', default=None,
                        help='Paths to evaluation result JSON files (optional)')
    parser.add_argument('--output_dir', type=str, default='experiments/comparison',
                        help='Output directory for comparison plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training histories
    print("Loading training histories...")
    training_histories = {}
    for exp in args.experiments:
        exp_path = Path(exp)
        if not exp_path.exists():
            exp_path = Path('experiments') / exp

        history = load_training_history(exp_path / 'checkpoints')
        training_histories[exp_path.name] = history

        if history:
            print(f"✓ Loaded {exp_path.name}")
        else:
            print(f"✗ Could not load {exp_path.name}")

    # Plot training curves
    if any(h is not None for h in training_histories.values()):
        print("\nPlotting training curves...")
        plot_training_curves(
            training_histories,
            save_path=output_dir / 'training_curves.png'
        )

    # Load and compare evaluation results
    if args.eval_results:
        print("\nLoading evaluation results...")
        eval_results = {}
        for i, result_path in enumerate(args.eval_results):
            results = load_evaluation_results(result_path)
            exp_name = args.experiments[i] if i < len(args.experiments) else f"Model {i+1}"
            eval_results[exp_name] = results

            if results:
                print(f"✓ Loaded {exp_name}")
            else:
                print(f"✗ Could not load {exp_name}")

        if any(r is not None for r in eval_results.values()):
            print("\nComparing metrics...")
            compare_metrics(
                eval_results,
                save_path=output_dir / 'metrics_comparison.png'
            )

    print(f"\n[checkmark.circle] Comparison complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
