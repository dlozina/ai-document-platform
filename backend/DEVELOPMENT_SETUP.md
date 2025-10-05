# Backend Development Setup Guide

This guide explains how to set up the development environment for the Python backend services with pre-commit hooks and CI/CD.

## Prerequisites

- Python 3.12+
- pip
- Git

## Quick Setup

### 1. Create Virtual Environment (IMPORTANT!)

**⚠️ CRITICAL: Always use a virtual environment to avoid messing up your local Python installation!**

```bash
cd backend
python -m venv venv-dev
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate
```

### 2. Install Development Dependencies

```bash
# Make sure you're in the virtual environment (you should see (venv-dev) in your prompt)
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
# Still in the virtual environment
pip install pre-commit
pre-commit install
```

### 4. Verify Installation

```bash
# Still in the virtual environment
pre-commit run --all-files
```

### 5. Deactivate When Done

```bash
deactivate  # Exit the virtual environment
```

## What's Included

### Pre-commit Hooks

The following hooks run automatically on every commit:

- **Code Formatting**: `black` - Automatic code formatting
- **Import Sorting**: `isort` - Organizes imports
- **Linting**: `ruff` - Fast Python linter
- **Type Checking**: `mypy` - Static type checking
- **Security Scanning**: `bandit` - Security vulnerability detection
- **Dependency Check**: `safety` - Checks for known vulnerabilities
- **General Checks**: Trailing whitespace, file endings, YAML/JSON validation

### GitHub Actions CI/CD

Three workflows run on every push and pull request:

1. **Backend Lint and Format Check** (`.github/workflows/backend-lint.yml`)
   - Runs all linting tools
   - Checks code formatting
   - Validates imports and types

2. **Backend Test Suite** (`.github/workflows/backend-test.yml`)
   - Runs tests for all 6 services in parallel
   - Generates coverage reports
   - Uploads test results

3. **Backend Security Scan** (`.github/workflows/backend-security.yml`)
   - Runs Bandit security scanner
   - Checks dependencies with Safety
   - Runs Semgrep security analysis
   - Comments on PRs with security findings

## Service-Specific Setup

Each service has its own dependencies. To work on a specific service:

```bash
cd backend/[service-name]
pip install -r requirements.txt
# or if using pyproject.toml:
pip install -e .[dev]
```

## Configuration Files

- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`pyproject.toml`** - Tool configurations (black, isort, ruff, mypy, bandit)
- **`requirements-dev.txt`** - Development dependencies

## Manual Commands

**⚠️ Always activate the virtual environment first:**
```bash
cd backend
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate
```

### Format Code
```bash
# Make sure you're in the virtual environment
black .
isort .
```

### Run Linting
```bash
# Make sure you're in the virtual environment
ruff check .
mypy .
```

### Run Security Scan
```bash
# Make sure you're in the virtual environment
bandit -r .
safety check
```

### Updating Dependencies
```bash
# Update vulnerable packages
pip install --upgrade sentence-transformers python-multipart

# Update requirements files with new versions
# (This should be done after upgrading packages)
# Example: sentence-transformers==2.7.0 → sentence-transformers==5.1.1
# Example: python-multipart==0.0.12 → python-multipart==0.0.20
```

### Run Tests
```bash
# Make sure you're in the virtual environment
cd [service-name]
pytest tests/ -v --cov=src
```

### Run Pre-commit on Specific Service
```bash
# Run on specific service directory
pre-commit run --files gateway/

# Run on specific files
pre-commit run --files gateway/src/main.py gateway/src/config.py

# Run specific hooks only
pre-commit run black isort --files gateway/

# Run with verbose output
pre-commit run --files gateway/ --verbose
```

## Troubleshooting

### Pre-commit Hooks Failing

**First, make sure you're in the virtual environment:**
```bash
cd backend
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate
```

1. **Fix formatting issues**:
   ```bash
   # Make sure you're in the virtual environment
   black .
   isort .
   ```

2. **Fix linting issues**:
   ```bash
   # Make sure you're in the virtual environment
   ruff check . --fix
   ```

3. **Skip hooks temporarily** (not recommended):
   ```bash
   git commit --no-verify -m "your message"
   ```

### Virtual Environment Issues

**If you accidentally installed packages globally:**
```bash
# Check what's installed globally
pip list

# Create a fresh virtual environment
cd backend
rm -rf venv-dev
python -m venv venv-dev
source venv-dev/bin/activate
pip install -r requirements-dev.txt
```

### CI/CD Failures

- Check the GitHub Actions logs for specific error messages
- Most failures are due to formatting or linting issues
- Security scan failures require code changes to fix vulnerabilities

## Adding New Services

When adding a new Python service:

1. Create the service directory under `backend/`
2. Add `requirements.txt` and/or `pyproject.toml`
3. Add tests in `tests/` directory
4. Update the GitHub Actions matrix in `backend-test.yml` to include the new service

## Frontend Integration

When the frontend is ready, create separate workflows:
- `.github/workflows/frontend-lint.yml`
- `.github/workflows/frontend-test.yml`
- `.github/workflows/frontend-security.yml`

This keeps Python and JavaScript tooling separate and focused.

## Virtual Environment Best Practices

### Why Use Virtual Environments?

- **Isolation**: Prevents conflicts between different projects
- **Clean System**: Keeps your global Python installation clean
- **Reproducibility**: Ensures consistent environments across team members
- **Safety**: Prevents accidentally breaking system Python

### Virtual Environment Management

**Creating a new environment:**
```bash
cd backend
python -m venv venv-dev
```

**Activating the environment:**
```bash
# Linux/macOS
source venv-dev/bin/activate

# Windows
venv-dev\Scripts\activate
```

**Deactivating the environment:**
```bash
deactivate
```

**Checking if you're in a virtual environment:**
- Your prompt should show `(venv-dev)` at the beginning
- Run `which python` - it should point to your venv directory

### Common Mistakes to Avoid

1. **Installing packages globally** - Always activate venv first
2. **Forgetting to activate venv** - Check your prompt for `(venv-dev)`
3. **Committing venv to git** - Virtual environments are in `.gitignore`
4. **Committing cache directories** - `.mypy_cache/`, `.ruff_cache/`, etc. are in `.gitignore`
5. **Using different Python versions** - Make sure you're using Python 3.12+
