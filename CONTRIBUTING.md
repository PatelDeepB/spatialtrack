# Contributing to SpatialTrack

Thank you for your interest in contributing to SpatialTrack! We welcome contributions from developers, researchers, and computer vision practitioners.

---

## 1. Development Environment Setup

### Prerequisites
- Python 3.11+
- Git

### Clone and Install
```bash
git clone https://github.com/PatelDeepB/spatialtrack.git
cd spatialtrack

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev,app]"

# Download and quantize default model weights
python scripts/download_model.py
```

---

## 2. Code Quality Standards

All pull requests must satisfy our quality and style guidelines:

1. **Functions & Methods:**
   - Single responsibility principle.
   - Functions must stay under 40 lines. Extract helpers if needed.
   - Max 3 to 4 parameters; pass configuration dataclasses or typed objects otherwise.
2. **File Size:**
   - Maximum 300 lines per source code file. Refactor into modular components if larger.
3. **Type Safety:**
   - 100% strict type annotations. No untyped dictionaries or `Any` where domain types apply.
   - MyPy must pass with zero warnings in strict mode.
4. **Testing:**
   - Write tests alongside new features.
   - Follow the AAA (Arrange, Act, Assert) testing pattern.
   - Target 85%+ code coverage.

---

## 3. Pre-Commit Quality Checks

Before submitting a pull request, run the verification suite:

```bash
# 1. Format code
ruff format src tests app

# 2. Check linter rules
ruff check src tests app

# 3. Static type analysis
mypy app src tests

# 4. Run test suite
pytest --cov=spatialtrack
```

---

## 4. Git Commit Guidelines

We enforce the Conventional Commits specification:

```
<type>(<scope>): <short description>
```

**Allowed Types:**
- `feat`: A new user-facing feature
- `fix`: A bug fix
- `refactor`: Code reorganization with no functional changes
- `docs`: Documentation updates
- `test`: Adding or updating test cases
- `chore`: Dependency updates or build adjustments
- `perf`: Performance optimizations

**Branch Naming:**
- `feat/feature-name`
- `fix/bug-description`
- `refactor/component-name`

---

## 5. Pull Request Checklist

- [ ] Code passes `ruff check` and `ruff format --check`.
- [ ] Code passes `mypy app src tests` with zero errors.
- [ ] All unit and integration tests pass via `pytest`.
- [ ] New features include corresponding unit tests.
