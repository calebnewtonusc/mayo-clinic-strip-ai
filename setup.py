"""Setup script for Mayo Clinic STRIP AI package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
# Filter out comments and empty lines
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="mayo-strip-ai",
    version="0.9.0",
    author="Mayo Clinic STRIP AI Team",
    author_email="",
    description="Production-ready deep learning system for stroke blood clot classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calebnewtonusc/mayo-clinic-strip-ai",
    project_urls={
        "Bug Tracker": "https://github.com/calebnewtonusc/mayo-clinic-strip-ai/issues",
        "Documentation": "https://github.com/calebnewtonusc/mayo-clinic-strip-ai/blob/main/README.md",
        "Source Code": "https://github.com/calebnewtonusc/mayo-clinic-strip-ai",
    },
    packages=find_packages(where=".", include=["src*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "deployment": [
            "Flask>=2.3.0",
            "Flask-CORS>=4.0.0",
            "gunicorn>=20.1.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.22.0",
            "ipywidgets>=8.0.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "mlflow>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mayo-strip-train=train:main",
            "mayo-strip-eval=evaluate:main",
            "mayo-strip-api=deploy.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "deep-learning",
        "medical-imaging",
        "stroke-classification",
        "pytorch",
        "computer-vision",
        "healthcare",
        "ai",
        "machine-learning",
        "clinical-ml",
    ],
    license="MIT",
    zip_safe=False,
)
