"""
Library Version Checker

Run this to check if your libraries need updates.
Recommended versions for compatibility:
- PyTorch >= 1.10.0
- segmentation_models_pytorch >= 0.3.0
- albumentations >= 1.3.0
- opencv-python >= 4.5.0
"""

import sys


def check_version(package_name, min_version=None):
    """Check if package is installed and print version."""
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"  WARNING: Version {version} < recommended {min_version}")
                return False
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"✗ {package_name}: Error - {e}")
        return False


def main():
    print("=" * 60)
    print("Checking library versions...")
    print("=" * 60)

    packages = {
        'torch': '1.10.0',
        'cv2': '4.5.0',  # opencv-python
        'numpy': '1.19.0',
        'albumentations': '1.3.0',
        'segmentation_models_pytorch': '0.3.0',
        'tqdm': None,
    }

    all_ok = True
    for package, min_ver in packages.items():
        if not check_version(package, min_ver):
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All libraries installed and up to date!")
    else:
        print("Some libraries need attention. See above.")
        print("\nTo update, run:")
        print("  pip install --upgrade torch opencv-python numpy albumentations segmentation-models-pytorch tqdm")
    print("=" * 60)


if __name__ == '__main__':
    main()