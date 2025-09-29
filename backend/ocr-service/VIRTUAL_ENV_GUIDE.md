# Virtual Environment Setup Guide

## Why Use a Virtual Environment?

Virtual environments are essential for Python development because they:
- **Isolate dependencies** - Keep project packages separate from system packages
- **Prevent conflicts** - Avoid version conflicts between different projects
- **Ensure reproducibility** - Same package versions across different machines
- **Easy cleanup** - Remove entire environment without affecting system

## Setting Up Virtual Environment

### Method 1: Using `venv` (Built-in, Recommended)

```bash
# Navigate to your OCR service directory
cd /Users/dlozina/workspace/assignment/abysalto/backend/ocr-service

# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Verify you're in the virtual environment
which python  # Should show path to venv/bin/python
python --version  # Should show Python 3.12+

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Note:
Install system dependencies (if not already installed)
# On macOS:
brew install tesseract poppler

# On Windows:
Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
Install Poppler from: http://blog.alivate.com.au/poppler-windows/
Add to PATH

# To deactivate when done
deactivate
```

### Method 2: Using `conda` (If you have Anaconda/Miniconda)

```bash
# Create conda environment
conda create -n ocr-service python=3.12

# Activate environment
conda activate ocr-service

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
conda deactivate
```

### Method 3: Using `pipenv` (Alternative)

```bash
# Install pipenv first
pip install pipenv

# Create Pipfile and install dependencies
pipenv install

# Activate environment
pipenv shell

# Install from requirements.txt
pipenv install -r requirements.txt
```

## Complete Setup Workflow

Here's the complete step-by-step process:

```bash
# 1. Navigate to project directory
cd /Users/dlozina/workspace/assignment/abysalto/backend/ocr-service

# 2. Create virtual environment
python3.12 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip to latest version
pip install --upgrade pip

# 5. Install system dependencies (if not already installed)
# On macOS:
brew install tesseract poppler

# On Ubuntu/Debian:
# sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils

# 6. Install Python dependencies
pip install -r requirements.txt

# 7. Verify installation
pip list

# 8. Test the service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 9. Run tests
pytest tests/ -v

# 10. When done, deactivate
deactivate
```

## Virtual Environment Best Practices

### 1. Always Activate Before Working
```bash
# Always activate before installing packages or running code
source venv/bin/activate
pip install some-package
python src/main.py
```

### 2. Add venv to .gitignore
```bash
# Add to .gitignore to avoid committing virtual environment
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

### 3. Create requirements.txt from Environment
```bash
# Generate requirements.txt from current environment
pip freeze > requirements.txt

# Or generate requirements with versions
pip freeze > requirements-lock.txt
```

### 4. Environment Variables
```bash
# Create .env file for environment variables
cp env.example .env

# Edit .env with your settings
nano .env
```

## Troubleshooting Virtual Environment Issues

### Issue 1: Python Version Not Found
```bash
# Check available Python versions
ls /usr/bin/python*

# Install Python 3.12 if needed
# On macOS:
brew install python@3.12

# On Ubuntu:
sudo apt-get install python3.12 python3.12-venv
```

### Issue 2: Permission Denied
```bash
# Fix permissions
chmod +x venv/bin/activate

# Or recreate environment
rm -rf venv
python3.12 -m venv venv
```

### Issue 3: Packages Not Found After Installation
```bash
# Verify virtual environment is active
which python
which pip

# Should show paths in venv directory
# If not, reactivate:
source venv/bin/activate
```

### Issue 4: Import Errors
```bash
# Check if packages are installed
pip list

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

## IDE Integration

### VS Code
1. Open project in VS Code
2. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Python: Select Interpreter"
4. Choose the interpreter from your virtual environment: `./venv/bin/python`

### PyCharm
1. Go to File → Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Choose "Existing Environment"
4. Select `./venv/bin/python`

## Quick Commands Reference

```bash
# Create and activate environment
python3.12 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run service
python -m uvicorn src.main:app --reload

# Run tests
pytest tests/ -v

# Deactivate
deactivate
```

## Production Deployment

For production, you can either:
1. **Use Docker** (recommended) - No virtual environment needed
2. **Use system Python** with proper dependency management
3. **Use virtual environment** on production server

The Docker approach is recommended as it handles all dependencies automatically.
