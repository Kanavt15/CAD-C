# 🤝 Contributing to LUNA16 Lung Cancer Detection

First off, thank you for considering contributing! This project aims to improve early lung cancer detection through deep learning, and every contribution helps save lives.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)

---

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for everyone, regardless of background or identity.

### Expected Behavior
- ✅ Be respectful and constructive
- ✅ Welcome newcomers and help them learn
- ✅ Focus on what's best for the project
- ✅ Show empathy towards others

### Unacceptable Behavior
- ❌ Harassment, discrimination, or trolling
- ❌ Personal attacks or insults
- ❌ Publishing others' private information
- ❌ Spam or off-topic discussions

---

## 🎯 How Can I Contribute?

### 1. Reporting Bugs 🐛
Found a bug? Help us fix it:

**Before submitting:**
- Check if the bug was already reported in [Issues](https://github.com/Kanavt15/CAD-C/issues)
- Try to reproduce with the latest version
- Gather relevant information (Python version, GPU/CPU, error messages)

**When reporting:**
```markdown
**Bug Description**: Clear description of the issue

**To Reproduce**:
1. Step 1
2. Step 2
3. See error

**Expected Behavior**: What should happen

**Environment**:
- OS: Windows 10 / Ubuntu 22.04 / macOS 13
- Python: 3.10
- PyTorch: 2.0.0
- GPU: NVIDIA RTX 3060 / CPU only

**Error Messages**:
```
[Paste error message here]
```

**Screenshots**: (if applicable)
```

### 2. Suggesting Enhancements 💡
Have an idea? We'd love to hear it:

**Enhancement categories:**
- 🏗️ New model architectures (DenseNet, MobileNet, etc.)
- 🎨 Visualization improvements (Grad-CAM, attention maps)
- ⚡ Performance optimizations
- 📊 New evaluation metrics
- 🌐 Web interface or API
- 📚 Documentation improvements

**Suggestion template:**
```markdown
**Feature Description**: What would you like to see?

**Use Case**: Why is this useful?

**Proposed Implementation**: How could this be implemented?

**Alternatives Considered**: Other approaches?
```

### 3. Contributing Code 💻
Ready to code? Awesome!

**Good first issues:**
- Documentation improvements
- Adding type hints
- Writing unit tests
- Fixing small bugs
- Adding new augmentation techniques

**Advanced contributions:**
- New CNN architectures
- 3D CNN implementation
- Web interface (Streamlit/Flask)
- Docker containerization
- Hyperparameter tuning
- Cross-validation

---

## 🛠️ Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/CAD-C.git
cd CAD-C

# Add upstream remote
git remote add upstream https://github.com/Kanavt15/CAD-C.git
```

### 2. Create Virtual Environment
```bash
python -m venv venv_dev
# Windows
venv_dev\Scripts\activate
# Linux/Mac
source venv_dev/bin/activate
```

### 3. Install Dependencies
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 isort pytest pytest-cov
```

### 4. Create Feature Branch
```bash
# Always branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

---

## 🔄 Pull Request Process

### 1. Before You Start
- [ ] Check existing issues and PRs
- [ ] Discuss major changes first (create an issue)
- [ ] Read the documentation

### 2. Making Changes
```bash
# Make your changes
git add .
git commit -m "Add feature: your feature description"

# Keep your fork updated
git fetch upstream
git rebase upstream/main

# Push to your fork
git push origin feature/your-feature-name
```

### 3. Creating the Pull Request

**PR Title Format:**
```
[Type] Brief description

Types: Fix, Feature, Docs, Refactor, Test, Style
```

**Examples:**
- `[Fix] Resolve CUDA out of memory error`
- `[Feature] Add DenseNet-121 architecture`
- `[Docs] Update installation instructions`
- `[Refactor] Improve data loading pipeline`

**PR Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that would break existing functionality)
- [ ] Documentation update

## Related Issue
Fixes #123

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Tested on CPU
- [ ] Tested on GPU
- [ ] Added unit tests
- [ ] All existing tests pass

## Screenshots (if applicable)
[Add screenshots here]

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
```

### 4. Review Process
- Maintainers will review within 1-3 days
- Address feedback and update PR
- Once approved, your PR will be merged!

---

## 📝 Coding Standards

### Python Style Guide
Follow [PEP 8](https://pep8.org/) with these specifics:

**Formatting:**
```python
# Use black for formatting (line length 100)
black --line-length 100 your_file.py

# Use isort for imports
isort --profile black your_file.py
```

**Naming Conventions:**
```python
# Variables and functions: snake_case
def load_ct_scan(series_uid):
    ct_array = sitk.GetArrayFromImage(ct_scan)
    return ct_array

# Classes: PascalCase
class ResNet101LungCancer(nn.Module):
    pass

# Constants: UPPER_CASE
PATCH_SIZE = 64
NUM_SLICES = 3
```

**Docstrings:**
```python
def extract_2d_patch(ct_array, center_z, center_y, center_x, patch_size=64):
    """
    Extract 2D multi-slice patch from CT scan.
    
    Args:
        ct_array (np.ndarray): CT scan array (Z, Y, X)
        center_z (int): Center Z coordinate
        center_y (int): Center Y coordinate
        center_x (int): Center X coordinate
        patch_size (int, optional): Patch size in pixels. Defaults to 64.
    
    Returns:
        np.ndarray: Extracted patch (num_slices, patch_size, patch_size)
        None: If patch is out of bounds
    
    Example:
        >>> patch = extract_2d_patch(ct_array, 100, 200, 150, patch_size=64)
        >>> print(patch.shape)  # (3, 64, 64)
    """
    # Implementation
    pass
```

**Type Hints:**
```python
from typing import Optional, Tuple, List
import numpy as np

def process_coordinates(
    coords: Tuple[float, float, float],
    origin: List[float],
    spacing: List[float]
) -> Optional[Tuple[int, int, int]]:
    """Process world coordinates to voxel coordinates."""
    pass
```

### Code Quality Tools
```bash
# Run before committing
black --check --line-length 100 .
isort --check-only --profile black .
flake8 --max-line-length 100 .
```

---

## 🧪 Testing Guidelines

### Writing Tests
```python
# tests/test_utils.py
import pytest
import numpy as np
from utils import normalize_hu, extract_2d_patch

def test_normalize_hu():
    """Test HU normalization."""
    image = np.array([-1000, 0, 400, 1000])
    normalized = normalize_hu(image)
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert normalized.dtype == np.float32

def test_extract_2d_patch_valid():
    """Test patch extraction with valid coordinates."""
    ct_array = np.random.randn(200, 512, 512)
    patch = extract_2d_patch(ct_array, 100, 256, 256, patch_size=64)
    assert patch is not None
    assert patch.shape == (3, 64, 64)

def test_extract_2d_patch_out_of_bounds():
    """Test patch extraction with out-of-bounds coordinates."""
    ct_array = np.random.randn(200, 512, 512)
    patch = extract_2d_patch(ct_array, 10, 10, 10, patch_size=64)
    assert patch is None
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_normalize_hu
```

---

## 📚 Documentation Standards

### Code Documentation
- Add docstrings to all public functions and classes
- Include type hints for function parameters and returns
- Provide usage examples in docstrings
- Comment complex algorithms

### README/Guides
- Use clear, concise language
- Include code examples
- Add screenshots where helpful
- Keep formatting consistent

### Jupyter Notebooks
- Add markdown cells explaining each section
- Include cell outputs for reference
- Document hyperparameters and configurations
- Add visualizations and results

---

## 🎨 Contribution Ideas

### Easy (Good First Issues)
- [ ] Add more data augmentation techniques
- [ ] Improve error messages
- [ ] Add progress bars to long-running functions
- [ ] Fix typos in documentation
- [ ] Add type hints to existing code
- [ ] Create unit tests for utility functions

### Medium
- [ ] Implement DenseNet or MobileNet architecture
- [ ] Add Grad-CAM visualization
- [ ] Create batch inference script
- [ ] Implement k-fold cross-validation
- [ ] Add DICOM support improvements
- [ ] Create REST API with FastAPI

### Advanced
- [ ] Implement 3D CNN architecture
- [ ] Add model interpretability (attention maps, saliency maps)
- [ ] Create web interface with Streamlit
- [ ] Docker containerization
- [ ] Add hyperparameter optimization (Optuna)
- [ ] Implement federated learning
- [ ] Create mobile app (TensorFlow Lite/ONNX)

---

## 🏆 Recognition

Contributors will be:
- Listed in README.md Contributors section
- Mentioned in release notes
- Given credit in academic papers (if applicable)

---

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: [your.email@example.com](mailto:your.email@example.com)

### Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [LUNA16 Dataset](https://luna16.grand-challenge.org/)
- [SimpleITK Guide](https://simpleitk.readthedocs.io/)
- [Project README](README.md)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<div align="center">

### 🌟 Thank You for Contributing!

*Every contribution, no matter how small, makes a difference.*

**Together, we can improve early cancer detection and save lives.**

</div>
