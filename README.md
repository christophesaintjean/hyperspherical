# hyperspherical

> Hyperspherical neural network layers for PyTorch.

![CI main](https://github.com/christophesaintjean/hyperspherical/actions/workflows/ci.yml/badge.svg?branch=main)
![CI develop](https://github.com/christophesaintjean/hyperspherical/actions/workflows/ci.yml/badge.svg?branch=develop)

`hyperspherical` is a Python library extending PyTorch with layers, losses, and tools defined on **hyperspherical spaces
**.  
Inspired by `torch.nn`, it provides modules such as `Spherical` and `Conv2dSpherical`, allowing you to build networks
better suited for detection, classification, or representation tasks.

---

## 🚀 Installation

From PyPI (coming soon):

```bash
pip install hyperspherical
```

From source with pip:

```bash
git clone https://github.com/christophesaintjean/hyperspherical.git
cd hyperspherical
pip install .
```

From source with uv:

```bash
git clone https://github.com/christophesaintjean/hyperspherical.git
cd hyperspherical
uv pip install .
```

## Tests

Tests use `pytest. To run the test suite:

```bash
pytest
```

To measure code coverage:

```bash
pytest --cov=hyperspherical --cov-branch --cov-report=html
```

HTML reports open in `htmlcov/index.html`.

## Contributing

Thanks for contributing to hyperspherical! Here's a guide for getting started, even if you are new to Git or GitHub.

1. Clone the project and create a virtual environment

```bash
git clone git@github.com:christophesaintjean/hyperspherical.git
cd hyperspherical
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

2. Branches and workflow
    * **main**: stable versions only. Protected by CI and review.
    * **develop**: development branch. All tests must pass before merge.
    * **feature/xxx**: for each new feature or bugfix.

3. Pull Request
    * Create a new branch from `develop` for each feature or bugfix.
    ```bash
   git checkout develop
   git checkout -b feature/your-feature-name
    ```
    * Commit messages should be clear and descriptive.
   ``` bash
    git add <files>
    git commit -m "Add a clear and descriptive message"
    git push origin feature/your-feature-name
    * Make sure all tests pass before pushing.
    * Open a Pull Request against `develop`.
    * At least one approval is required before merging.

4. Tests
    * Add tests for new features or bug fixes in the `tests/` directory.
    * Ensure all tests pass before pushing changes.
    * Example for a `spherical_init` initialization test:
    ```python
   import pytest
   import torch
   from hyperspherical.initializers import kmeans_
   
   @pytest.mark.parametrize("n_clusters, n_samples", [(5, 200), (10, 500)])
   def test_kmeans_number_of_clusters(n_clusters, n_samples):
    """
    Test that the number of clusters in kmeans_ matches the expected number.
    :param n_clusters: number of clusters
    :param n_samples: number of samples
    :return: None
    """
    data = torch.randn(n_samples, 4)
    spheres = torch.empty(n_clusters, 5)
    _ = kmeans_(spheres, data)
    assert spheres.size(0) == n_clusters
    ````
   Run tests with:
    ```bash
    pytest
    ```
5. Best Practices and code style
    * Follow PEP 8 guidelines for Python code.
    * Use type hints where appropriate.
    * Write clear and concise docstrings for functions and classes.
    * Use meaningful variable and function names.
    * Keep functions and classes focused on a single responsibility.
    * Respect **code formatting** using `black`, `isort`, and `ruff`:
    ```bash
    black .
    isort .
    ruff check .
    ```

Thanks for contributing and helping hyperspherical grow!
