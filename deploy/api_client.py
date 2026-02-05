"""Simple client for testing the inference API."""

import requests
import argparse
from pathlib import Path
import json


def predict_image(api_url, image_path, uncertainty=False):
    """Send image for prediction.

    Args:
        api_url: Base URL of API
        image_path: Path to image file
        uncertainty: Whether to compute uncertainty

    Returns:
        Prediction results
    """
    endpoint = f"{api_url}/predict"

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'uncertainty': str(uncertainty).lower()}

        response = requests.post(endpoint, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def batch_predict_images(api_url, image_paths):
    """Send multiple images for prediction.

    Args:
        api_url: Base URL of API
        image_paths: List of image file paths

    Returns:
        Batch prediction results
    """
    endpoint = f"{api_url}/batch_predict"

    files = [('files', open(path, 'rb')) for path in image_paths]

    try:
        response = requests.post(endpoint, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    finally:
        # Close all files
        for _, f in files:
            f.close()


def get_model_info(api_url):
    """Get model information.

    Args:
        api_url: Base URL of API

    Returns:
        Model information
    """
    endpoint = f"{api_url}/model_info"
    response = requests.get(endpoint)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def health_check(api_url):
    """Check API health.

    Args:
        api_url: Base URL of API

    Returns:
        Health status
    """
    endpoint = f"{api_url}/health"
    response = requests.get(endpoint)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def main():
    parser = argparse.ArgumentParser(description='Test inference API')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000',
                        help='API base URL')
    parser.add_argument('--action', type=str, required=True,
                        choices=['health', 'info', 'predict', 'batch_predict'],
                        help='Action to perform')
    parser.add_argument('--image', type=str,
                        help='Path to image file (for predict)')
    parser.add_argument('--images', type=str, nargs='+',
                        help='Paths to image files (for batch_predict)')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Compute uncertainty (for predict)')
    args = parser.parse_args()

    try:
        if args.action == 'health':
            print("Checking API health...")
            result = health_check(args.api_url)
            print(json.dumps(result, indent=2))

        elif args.action == 'info':
            print("Getting model info...")
            result = get_model_info(args.api_url)
            print(json.dumps(result, indent=2))

        elif args.action == 'predict':
            if not args.image:
                print("Error: --image required for predict action")
                return

            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Error: Image not found: {args.image}")
                return

            print(f"Predicting image: {args.image}")
            result = predict_image(args.api_url, image_path, args.uncertainty)

            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("\nProbabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")

            if 'uncertainty' in result:
                print("\nUncertainty:")
                for class_name, unc in result['uncertainty'].items():
                    print(f"  {class_name}: {unc:.4f}")
            print("="*60)

        elif args.action == 'batch_predict':
            if not args.images:
                print("Error: --images required for batch_predict action")
                return

            # Check all images exist
            image_paths = [Path(img) for img in args.images]
            missing = [p for p in image_paths if not p.exists()]
            if missing:
                print(f"Error: Images not found: {missing}")
                return

            print(f"Predicting {len(image_paths)} images...")
            result = batch_predict_images(args.api_url, image_paths)

            print("\n" + "="*60)
            print("BATCH PREDICTION RESULTS")
            print("="*60)
            for r in result['results']:
                print(f"\n{r['filename']}:")
                print(f"  Class: {r['class_name']}")
                print(f"  Confidence: {r['confidence']:.4f}")
            print("="*60)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
