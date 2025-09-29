# Virtual Environment Setup Guide

This guide explains how to set up and manage Python virtual environments for the Embedding Service.

## Why Use Virtual Environments?

Virtual environments provide isolated Python environments for your project, preventing dependency conflicts and ensuring reproducible builds.

## Setup Methods

### Method 1: Using `python -m venv` (Recommended)

This is the standard Python approach and works on all platforms.

#### 1. Create Virtual Environment
```bash
# Navigate to the embedding service directory
cd backend/embedding-service

# Create virtual environment
python -m venv venv
```

#### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```cmd
venv\Scripts\activate
```

**On Windows PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

#### 3. Verify Activation
```bash
# Check Python path
which python  # Should point to venv/bin/python

# Check pip path
which pip     # Should point to venv/bin/pip
```

#### 4. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

#### 5. Deactivate When Done
```bash
deactivate
```

### Method 2: Using `conda`

If you prefer conda for environment management:

#### 1. Create Conda Environment
```bash
# Create environment with Python 3.12
conda create -n embedding-service python=3.12

# Activate environment
conda activate embedding-service
```

#### 2. Install Dependencies
```bash
# Install PyTorch (CPU version)
conda install pytorch cpuonly -c pytorch

# Install other dependencies via pip
pip install -r requirements.txt
```

#### 3. Deactivate
```bash
conda deactivate
```

### Method 3: Using `virtualenv`

Alternative virtual environment tool:

#### 1. Install virtualenv
```bash
pip install virtualenv
```

#### 2. Create Environment
```bash
virtualenv venv
```

#### 3. Activate and Install
```bash
# Activate (same as venv)
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure with Virtual Environment

```
backend/embedding-service/
├── venv/                    # Virtual environment (don't commit)
│   ├── bin/                 # Executables (macOS/Linux)
│   ├── lib/                 # Installed packages
│   └── pyvenv.cfg          # Environment config
├── src/                     # Source code
├── tests/                   # Test files
├── requirements.txt         # Dependencies
├── pyproject.toml          # Project configuration
└── .gitignore              # Git ignore rules
```

## Managing Dependencies

### Installing New Packages

#### 1. Activate Environment
```bash
source venv/bin/activate
```

#### 2. Install Package
```bash
pip install package-name
```

#### 3. Update Requirements
```bash
pip freeze > requirements.txt
```

### Development Dependencies

Install development tools:
```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support
- `pytest-cov` - Coverage reporting

### Optional Dependencies

Install optional features:
```bash
# For caching support
pip install -e ".[cache]"

# For monitoring
pip install -e ".[monitoring]"

# For document processing
pip install -e ".[documents]"

# All optional dependencies
pip install -e ".[cache,monitoring,documents]"
```

## Environment Management Best Practices

### 1. Always Use Virtual Environments
- Never install packages globally
- Each project should have its own environment
- Use descriptive environment names

### 2. Pin Dependencies
- Use `requirements.txt` for exact versions
- Use `pyproject.toml` for flexible version ranges
- Regularly update dependencies

### 3. Environment Isolation
- Don't commit `venv/` directory
- Use `.gitignore` to exclude virtual environments
- Document setup process in README

### 4. Reproducible Builds
```bash
# Generate exact requirements
pip freeze > requirements-exact.txt

# Install from exact requirements
pip install -r requirements-exact.txt
```

## Troubleshooting Virtual Environments

### Common Issues

#### 1. Activation Script Not Found
**Error**: `source: venv/bin/activate: No such file or directory`

**Solution**:
```bash
# Check if venv was created properly
ls -la venv/

# Recreate if necessary
rm -rf venv
python -m venv venv
```

#### 2. Wrong Python Version
**Error**: Using system Python instead of venv Python

**Solution**:
```bash
# Check which Python is being used
which python

# Should show: /path/to/project/venv/bin/python

# If not, reactivate
deactivate
source venv/bin/activate
```

#### 3. Package Installation Issues
**Error**: `Permission denied` or `No module named pip`

**Solution**:
```bash
# Ensure venv is activated
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install packages
pip install -r requirements.txt
```

#### 4. Environment Variables Not Set
**Error**: Environment variables not available in venv

**Solution**:
```bash
# Set environment variables before activation
export QDRANT_HOST=localhost
source venv/bin/activate

# Or use .env file
pip install python-dotenv
# Load in your application
```

### Platform-Specific Issues

#### Windows PowerShell Execution Policy
**Error**: `execution of scripts is disabled on this system`

**Solution**:
```powershell
# Set execution policy (run as administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use Command Prompt instead
cmd
venv\Scripts\activate
```

#### macOS Security Warnings
**Error**: `"activate" cannot be opened because it is from an unidentified developer`

**Solution**:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine venv/bin/activate

# Or allow in System Preferences > Security & Privacy
```

## IDE Integration

### VS Code
1. Open project folder in VS Code
2. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose `./venv/bin/python` (or `./venv/Scripts/python.exe` on Windows)

### PyCharm
1. Open project in PyCharm
2. Go to Settings → Project → Python Interpreter
3. Add interpreter → Existing environment
4. Select `venv/bin/python`

### Jupyter Notebook
```bash
# Install jupyter in venv
pip install jupyter

# Install kernel
python -m ipykernel install --user --name=embedding-service --display-name="Embedding Service"

# Start jupyter
jupyter notebook
```

## Production Considerations

### 1. Environment Variables
```bash
# Use environment files
cp env.example .env

# Or set system environment variables
export QDRANT_HOST=production-qdrant.com
export QDRANT_API_KEY=your-api-key
```

### 2. Dependency Management
```bash
# Use exact versions for production
pip freeze > requirements-production.txt

# Install in production
pip install -r requirements-production.txt
```

### 3. Container Deployment
```dockerfile
# Use virtual environment in Docker
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt

COPY src/ ./src/
CMD ["venv/bin/python", "-m", "uvicorn", "src.main:app"]
```

## Cleanup

### Remove Virtual Environment
```bash
# Deactivate first
deactivate

# Remove directory
rm -rf venv
```

### Clean Package Cache
```bash
# Clear pip cache
pip cache purge

# Clear conda cache
conda clean --all
```

## Summary

Virtual environments are essential for Python development. They provide:

- **Isolation**: Separate dependencies per project
- **Reproducibility**: Consistent environments across machines
- **Flexibility**: Easy switching between Python versions
- **Safety**: No conflicts with system packages

Follow these practices for successful virtual environment management:

1. Always use virtual environments for Python projects
2. Pin your dependencies in `requirements.txt`
3. Document your setup process
4. Use version control to track configuration files
5. Test your setup on clean systems regularly

For the Embedding Service specifically:
- Use Python 3.12+ for best performance
- Install PyTorch CPU version for compatibility
- Consider GPU support for production workloads
- Monitor memory usage with large models
