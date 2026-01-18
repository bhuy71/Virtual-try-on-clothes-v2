from setuptools import setup, find_packages

setup(
    name="virtual_tryon",
    version="0.1.0",
    description="Virtual Try-On: Traditional CP-VTON and SOTA HR-VITON Methods",
    author="Virtual Try-On Research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "demo": [
            "gradio>=3.50.0",
        ],
    },
)
