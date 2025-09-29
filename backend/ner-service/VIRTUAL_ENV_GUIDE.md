# Virtual Environment Guide for NER Service

This guide explains how to set up and manage virtual environments for the NER Service development.

## Why Use Virtual Environments?

Virtual environments isolate your project dependencies from your system Python installation, preventing conflicts and ensuring reproducible builds.

## Quick Start

### 1. Create Virtual Environment

```bash
# Navigate to the NER service directory
cd backend/ner-service

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install spaCy models on macOS with Apple Silicon
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# Install spaCy on other platforms
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

```

### 3. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test spaCy models
python -c "import spacy; print(spacy.load('en_core_web_sm'))"
```

## Detailed Setup

### Method 1: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models on macOS with Apple Silicon
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# Download spaCy models on other platforms
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

### Method 2: Using conda

```bash
# Create conda environment
conda create -n ner-service python=3.12

# Activate environment
conda activate ner-service

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

### Method 3: Using pipenv

```bash
# Install pipenv if not already installed
pip install pipenv

# Create Pipfile and install dependencies
pipenv install

# Activate virtual environment
pipenv shell

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## Environment Management

### Activating and Deactivating

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate virtual environment
deactivate
```

### Checking Environment Status

```bash
# Check if virtual environment is active
echo $VIRTUAL_ENV  # macOS/Linux
echo %VIRTUAL_ENV% # Windows

# Check Python location
which python  # macOS/Linux
where python  # Windows

# Check installed packages
pip list
```

### Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade spacy

# Update spaCy models
python -m spacy download en_core_web_sm --upgrade
```

## Development Workflow

### 1. Daily Development

```bash
# Activate virtual environment
source venv/bin/activate

# Run the service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload

# Run tests
pytest tests/ -v

# Deactivate when done
deactivate
```

### 2. Adding New Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install new package
pip install new-package

# Update requirements.txt
pip freeze > requirements.txt

# Commit changes
git add requirements.txt
git commit -m "Add new-package dependency"
```

### 3. Sharing Environment

```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Share with team
git add requirements.txt
git commit -m "Update dependencies"
git push
```

## Troubleshooting

### Common Issues

1. **Virtual environment not activating**
   ```bash
   # Check if venv directory exists
   ls -la venv/
   
   # Recreate if needed
   rm -rf venv
   python -m venv venv
   ```

2. **spaCy models not found**
   ```bash
   # Check if models are installed
   python -c "import spacy; spacy.util.get_package_path('en_core_web_sm')"
   
   # Reinstall models
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

3. **Permission errors**
   ```bash
   # Fix permissions (macOS/Linux)
   chmod +x venv/bin/activate
   
   # Or recreate with proper permissions
   rm -rf venv
   python -m venv venv
   ```

4. **Package conflicts**
   ```bash
   # Check for conflicts
   pip check
   
   # Reinstall if needed
   pip install --force-reinstall -r requirements.txt
   ```

### Environment Validation

```bash
# Check Python version
python --version  # Should be 3.12+

# Check pip version
pip --version

# Check virtual environment
echo $VIRTUAL_ENV  # Should show path to venv

# Test imports
python -c "import fastapi, spacy, uvicorn; print('All imports successful')"

# Test spaCy models
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully')"
```

## IDE Integration

### VS Code

1. **Select Python Interpreter**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your virtual environment

2. **Configure Settings**
   ```json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.terminal.activateEnvironment": true
   }
   ```

### PyCharm

1. **Configure Project Interpreter**
   - Go to File → Settings → Project → Python Interpreter
   - Click the gear icon → Add
   - Choose "Existing Environment"
   - Select `venv/bin/python`

### Jupyter Notebook

```bash
# Install jupyter in virtual environment
pip install jupyter

# Start jupyter
jupyter notebook

# Or use jupyter lab
pip install jupyterlab
jupyter lab
```

## Production Considerations

### 1. Environment Variables

```bash
# Create production environment file
cp env.example .env.production

# Set production values
echo "LOG_LEVEL=WARNING" >> .env.production
echo "SPACY_MODEL=en_core_web_lg" >> .env.production
```

### 2. Docker Integration

```dockerfile
# Use virtual environment in Docker
FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg

# Copy application
COPY src/ ./src/

# Run application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 3. CI/CD Integration

```yaml
# GitHub Actions example
name: Test NER Service

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    
    - name: Create virtual environment
      run: python -m venv venv
    
    - name: Activate virtual environment
      run: source venv/bin/activate
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
    
    - name: Run tests
      run: pytest tests/ -v
```

## Best Practices

### 1. Environment Isolation

- Always use virtual environments for projects
- Never install packages globally
- Keep requirements.txt up to date

### 2. Dependency Management

```bash
# Pin specific versions
pip install spacy==3.7.2

# Use requirements.txt for reproducible builds
pip freeze > requirements.txt

# Use requirements-dev.txt for development dependencies
pip install pytest pytest-asyncio
pip freeze > requirements-dev.txt
```

### 3. Environment Documentation

```bash
# Document Python version
echo "Python 3.12.0" > .python-version

# Document dependencies
pip freeze > requirements.txt

# Document setup steps
echo "1. Create virtual environment: python -m venv venv" > SETUP.md
echo "2. Activate: source venv/bin/activate" >> SETUP.md
echo "3. Install: pip install -r requirements.txt" >> SETUP.md
```

### 4. Regular Maintenance

```bash
# Weekly updates
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Monthly security updates
pip install --upgrade --security -r requirements.txt

# Quarterly major updates
pip install --upgrade spacy fastapi uvicorn
```

## Advanced Usage

### 1. Multiple Environments

```bash
# Development environment
python -m venv venv-dev
source venv-dev/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Production environment
python -m venv venv-prod
source venv-prod/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
# Set environment-specific variables
export SPACY_MODEL=en_core_web_sm
export LOG_LEVEL=DEBUG

# Or use .env file
echo "SPACY_MODEL=en_core_web_sm" > .env
echo "LOG_LEVEL=DEBUG" >> .env
```

### 3. Scripts and Automation

```bash
# Create activation script
cat > activate_dev.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export SPACY_MODEL=en_core_web_sm
export LOG_LEVEL=DEBUG
echo "Development environment activated"
EOF

chmod +x activate_dev.sh
```

This guide should help you set up and manage virtual environments effectively for the NER Service development.
