"""Grad-CAM and Grad-CAM++ implementations for model interpretability."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List


class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping.

    Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (2017)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize Grad-CAM.

        Args:
            model: The model to interpret
            target_layer: The layer to compute CAM from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            Grad-CAM heatmap as numpy array (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Calculate Grad-CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def __del__(self):
        """Remove hooks when object is destroyed."""
        self.forward_hook.remove()
        self.backward_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++: Improved Grad-CAM with better localization.

    Reference: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based
    Visual Explanations for Deep Convolutional Networks" (2018)
    """

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            Grad-CAM++ heatmap as numpy array (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Calculate Grad-CAM++ weights
        gradients = self.gradients
        activations = self.activations

        # First derivative
        alpha_numer = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2)

        # Second derivative (approximation)
        alpha_denom += (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_numer / alpha_denom

        # Weight the gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlay heatmap on original image.

    Args:
        image: Original image (H, W, 3) in range [0, 255] or [0, 1]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use

    Returns:
        Overlayed image (H, W, 3) in range [0, 255]
    """
    # Normalize image to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to color
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlayed


def get_target_layer(model: torch.nn.Module, model_type: str = 'resnet') -> torch.nn.Module:
    """Get the target layer for Grad-CAM based on model architecture.

    Args:
        model: The model
        model_type: Type of model ('resnet', 'efficientnet', 'vit', etc.)

    Returns:
        Target layer module
    """
    if model_type == 'resnet':
        # For ResNet, use the last layer of layer4
        if hasattr(model, 'backbone'):
            return model.backbone.layer4[-1]
        else:
            return model.layer4[-1]

    elif model_type == 'efficientnet':
        # For EfficientNet, use the last conv layer
        if hasattr(model, 'backbone'):
            return model.backbone.features[-1]
        else:
            return model.features[-1]

    elif model_type == 'simple_cnn':
        # For SimpleCNN, use the last conv layer
        if hasattr(model, 'features'):
            return model.features[-4]  # Last conv before pooling

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_guided_backprop(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None
) -> np.ndarray:
    """Compute guided backpropagation.

    Args:
        model: The model
        input_tensor: Input image tensor (1, C, H, W)
        target_class: Target class index

    Returns:
        Guided backpropagation saliency map
    """
    # Store original ReLU forward functions
    relu_outputs = []

    def relu_hook_function(module, grad_in, grad_out):
        """Hook to modify ReLU backward pass."""
        return (torch.clamp(grad_in[0], min=0.0),)

    # Register hooks for all ReLU layers
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_full_backward_hook(relu_hook_function))

    # Set model to eval and require grad
    model.eval()
    input_tensor.requires_grad = True

    # Forward pass
    output = model(input_tensor)

    # Use predicted class if not specified
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients
    model.zero_grad()

    # Backward pass
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)

    # Get gradients
    gradients = input_tensor.grad.cpu().numpy()[0]

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Normalize for visualization
    gradients = np.maximum(gradients, 0)
    gradients = gradients.transpose(1, 2, 0)  # CHW to HWC

    # Convert to grayscale if RGB
    if gradients.shape[2] == 3:
        gradients = np.mean(gradients, axis=2)

    # Normalize to [0, 1]
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

    return gradients
