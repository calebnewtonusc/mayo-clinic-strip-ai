"""Feature visualization and analysis tools."""

import torch
import numpy as np
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with 'pip install umap-learn'")


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layer_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features from a specific layer.

    Args:
        model: The model
        dataloader: DataLoader for the dataset
        device: Device to run on
        layer_name: Name of layer to extract from (if None, uses penultimate layer)

    Returns:
        Tuple of (features, labels, patient_ids)
    """
    model.eval()
    features_list = []
    labels_list = []

    # Hook to capture features
    features_hook = []

    def hook_fn(module, input, output):
        features_hook.append(output.detach())

    # Register hook
    if layer_name:
        layer = dict(model.named_modules())[layer_name]
    else:
        # Get penultimate layer (before final classifier)
        modules = list(model.children())
        layer = modules[-2] if len(modules) > 1 else modules[-1]

    hook = layer.register_forward_hook(hook_fn)

    # Extract features
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Forward pass
            _ = model(images)

            # Get features from hook
            feats = features_hook[-1]

            # Flatten features
            feats = feats.view(feats.size(0), -1)

            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

            # Clear hook
            features_hook.clear()

    # Remove hook
    hook.remove()

    # Concatenate
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def compute_tsne(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> np.ndarray:
    """Compute t-SNE embedding.

    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of dimensions in embedding
        perplexity: t-SNE perplexity parameter
        random_state: Random seed

    Returns:
        Embedded features (n_samples, n_components)
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000
    )

    embedded = tsne.fit_transform(features)
    return embedded


def compute_umap(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """Compute UMAP embedding.

    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of dimensions in embedding
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed

    Returns:
        Embedded features (n_samples, n_components)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with 'pip install umap-learn'")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )

    embedded = reducer.fit_transform(features)
    return embedded


def compute_pca(
    features: np.ndarray,
    n_components: int = 2
) -> np.ndarray:
    """Compute PCA embedding.

    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of principal components

    Returns:
        Embedded features (n_samples, n_components)
    """
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(features)
    return embedded


def plot_embedding(
    embedded: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
    title: str = "Feature Embedding",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """Plot 2D embedding with class labels.

    Args:
        embedded: 2D embedded features (n_samples, 2)
        labels: Class labels (n_samples,)
        class_names: Names of classes
        title: Plot title
        save_path: Path to save plot (if None, shows plot)
        figsize: Figure size
    """
    if class_names is None:
        class_names = ['CE', 'LAA']

    plt.figure(figsize=figsize)

    # Plot each class with different color
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=[colors[i]],
            label=class_names[label] if label < len(class_names) else f"Class {label}",
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5,
            s=50
        )

    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_feature_separability(
    features: np.ndarray,
    labels: np.ndarray
) -> dict:
    """Analyze how well features separate classes.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels (n_samples,)

    Returns:
        Dictionary of separability metrics
    """
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    # Silhouette score
    silhouette = silhouette_score(features, labels)

    # Inter-class vs intra-class distances
    unique_labels = np.unique(labels)

    # Calculate centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = features[mask].mean(axis=0, keepdims=True)
        centroids.append(centroid)

    # Inter-class distance (distance between centroids)
    if len(centroids) == 2:
        inter_class_dist = np.linalg.norm(centroids[0] - centroids[1])
    else:
        inter_class_dist = np.mean(cdist(np.vstack(centroids), np.vstack(centroids)))

    # Intra-class distance (average distance to centroid within class)
    intra_class_dists = []
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_features = features[mask]
        distances = np.linalg.norm(class_features - centroids[i], axis=1)
        intra_class_dists.append(distances.mean())

    avg_intra_class_dist = np.mean(intra_class_dists)

    # Separability ratio (higher is better)
    separability_ratio = inter_class_dist / (avg_intra_class_dist + 1e-8)

    return {
        'silhouette_score': silhouette,
        'inter_class_distance': inter_class_dist,
        'avg_intra_class_distance': avg_intra_class_dist,
        'separability_ratio': separability_ratio
    }


def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
    n_features_to_plot: int = 10,
    save_path: Optional[str] = None
):
    """Plot distribution of top features by class.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels (n_samples,)
        class_names: Names of classes
        n_features_to_plot: Number of top features to plot
        save_path: Path to save plot
    """
    if class_names is None:
        class_names = ['CE', 'LAA']

    # Select top features by variance
    feature_var = np.var(features, axis=0)
    top_indices = np.argsort(feature_var)[-n_features_to_plot:]

    # Plot distributions
    n_cols = 2
    n_rows = (n_features_to_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]

        # Plot distribution for each class
        for label in np.unique(labels):
            mask = labels == label
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            ax.hist(
                features[mask, feat_idx],
                bins=30,
                alpha=0.6,
                label=class_name,
                density=True
            )

        ax.set_xlabel(f'Feature {feat_idx}', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_features_to_plot, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()
