# Docker Setup and Commands - Diabetes Prediction Pipeline

## Complete Docker Installation Guide

### Windows Installation

#### Step 1: System Requirements
- Windows 10/11 Pro, Enterprise, or Education (64-bit)
- OR Windows 10/11 Home with WSL 2
- At least 4GB RAM
- Virtualization enabled in BIOS

#### Step 2: Download and Install Docker Desktop
```bash
# Option 1: Download from official website
# Go to: https://docs.docker.com/desktop/install/windows-install/
# Download "Docker Desktop for Windows"

# Option 2: Using winget (Windows Package Manager)
winget install Docker.DockerDesktop

# Option 3: Using Chocolatey
choco install docker-desktop
```

#### Step 3: Installation Process
1. Run the installer as Administrator
2. Follow the installation wizard
3. Enable WSL 2 integration when prompted
4. Restart your computer when installation completes

#### Step 4: First Launch
1. Launch Docker Desktop from Start Menu
2. Accept the Docker Subscription Service Agreement
3. Wait for Docker Engine to start (whale icon in system tray)
4. Open Command Prompt or PowerShell and verify:
```bash
docker --version
docker-compose --version
```

### macOS Installation

#### Step 1: System Requirements
- macOS 10.15 or later
- At least 4GB RAM
- Apple chip (M1/M2) or Intel processor

#### Step 2: Download and Install Docker Desktop
```bash
# Option 1: Download from official website
# For Apple Silicon (M1/M2): https://docs.docker.com/desktop/install/mac-install/
# For Intel: https://docs.docker.com/desktop/install/mac-install/

# Option 2: Using Homebrew
brew install --cask docker

# Option 3: Using MacPorts
sudo port install docker-desktop
```

#### Step 3: Installation Process
1. Open the downloaded .dmg file
2. Drag Docker.app to Applications folder
3. Launch Docker from Applications
4. Grant necessary permissions when prompted
5. Wait for Docker to start

#### Step 4: Verify Installation
```bash
# Open Terminal and run:
docker --version
docker-compose --version
```

## Project Setup and Running

### Prerequisites Check
```bash
# Verify Docker is running (both Windows/Mac)
docker --version
docker-compose --version

# Check if Docker daemon is running
docker info
```

## Complete Project Setup and Running

### Step 1: Download/Clone Project
```bash
# If using Git (recommended)
git clone <your-repo-url>
cd Progetto_mott

# OR manually download and extract the project
# Navigate to the project folder in terminal/command prompt
```

### Step 2: Navigate to Project Directory

#### Windows (Command Prompt/PowerShell)
```cmd
cd "C:\Users\ludov\Desktop\Wild_Boy\Magistrale\primo anno\MOTT\Progetto_mott"
```

#### Windows (Git Bash)
```bash
cd "/c/Users/ludov/Desktop/Wild_Boy/Magistrale/primo anno/MOTT/Progetto_mott"
```

#### macOS/Linux (Terminal)
```bash
cd /path/to/your/Progetto_mott
# Example: cd ~/Downloads/Progetto_mott
```

### Step 3: Build the Docker Image

#### Windows
```cmd
# Using Command Prompt
docker build -t diabetes-pipeline:latest .

# Using PowerShell
docker build -t diabetes-pipeline:latest .
```

#### macOS/Linux
```bash
# Using Terminal
docker build -t diabetes-pipeline:latest .
```

### Step 4: Run the Project

#### Option A: Run Individual Services

##### Windows
```cmd
REM 1. Simple Test Suite (default)
docker run --rm -v "%cd%\outputs:/app/outputs" diabetes-pipeline:latest

REM 2. Clustering Analysis
docker run --rm -v "%cd%\outputs:/app/outputs" diabetes-pipeline:latest python metodi/cluster/clean_dataset_cluster.py

REM 3. Integrated System
docker run --rm -v "%cd%\outputs:/app/outputs" diabetes-pipeline:latest python metodi/terzo_metodo/hybrid_ml_clinical_rules_integrated.py
```

##### Windows PowerShell
```powershell
# 1. Simple Test Suite (default)
docker run --rm -v "${PWD}\outputs:/app/outputs" diabetes-pipeline:latest

# 2. Clustering Analysis
docker run --rm -v "${PWD}\outputs:/app/outputs" diabetes-pipeline:latest python metodi/cluster/clean_dataset_cluster.py

# 3. Integrated System
docker run --rm -v "${PWD}\outputs:/app/outputs" diabetes-pipeline:latest python metodi/terzo_metodo/hybrid_ml_clinical_rules_integrated.py
```

##### macOS/Linux
```bash
# 1. Simple Test Suite (default)
docker run --rm -v "$(pwd)/outputs:/app/outputs" diabetes-pipeline:latest

# 2. Clustering Analysis
docker run --rm -v "$(pwd)/outputs:/app/outputs" diabetes-pipeline:latest python metodi/cluster/clean_dataset_cluster.py

# 3. Integrated System
docker run --rm -v "$(pwd)/outputs:/app/outputs" diabetes-pipeline:latest python metodi/terzo_metodo/hybrid_ml_clinical_rules_integrated.py
```

#### Option B: Using Docker Compose (Recommended)

##### All Platforms (Windows/Mac/Linux)
```bash
# Build and run the full pipeline
docker-compose up

# Run in background (detached mode)
docker-compose up -d

# Run specific service
docker-compose up diabetes-pipeline
docker-compose up clustering-analysis
docker-compose up integrated-system
docker-compose up full-test-suite

# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Development Commands

## Advanced Usage and Development

### Interactive Container Development

#### Windows Command Prompt
```cmd
REM Start interactive bash session
docker run -it --rm -v "%cd%:/app" diabetes-pipeline:latest bash

REM Run specific Python command
docker run -it --rm -v "%cd%:/app" diabetes-pipeline:latest python -c "import pandas; print('Pandas version:', pandas.__version__)"
```

#### Windows PowerShell
```powershell
# Start interactive bash session
docker run -it --rm -v "${PWD}:/app" diabetes-pipeline:latest bash

# Run specific Python command
docker run -it --rm -v "${PWD}:/app" diabetes-pipeline:latest python -c "import pandas; print('Pandas version:', pandas.__version__)"
```

#### macOS/Linux Terminal
```bash
# Start interactive bash session
docker run -it --rm -v "$(pwd):/app" diabetes-pipeline:latest bash

# Run specific Python command
docker run -it --rm -v "$(pwd):/app" diabetes-pipeline:latest python -c "import pandas; print('Pandas version:', pandas.__version__)"
```

### Debugging
```bash
# Check container logs
docker logs diabetes-prediction-pipeline

# View running containers
docker ps

# View all containers
docker ps -a
```

## Volume Management

### Persistent Data
```bash
# Create named volume for outputs
docker volume create diabetes_outputs

# Run with named volume
docker run --rm -v diabetes_outputs:/app/outputs diabetes-pipeline:latest
```

## Production Deployment

### Multi-stage Pipeline
```bash
# Step 1: Run clustering comparison
docker-compose up clustering-analysis

# Step 2: Run integrated system with results
docker-compose up integrated-system

# Step 3: Verify with tests
docker-compose up full-test-suite
```

### Health Checks
```bash
# Test container health
docker run --rm diabetes-pipeline:latest python -c "import metodi.cluster.clean_dataset_cluster; print('System OK')"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Docker Not Running
**Error**: `Cannot connect to the Docker daemon`

**Solutions**:
- **Windows**: Launch Docker Desktop from Start Menu
- **macOS**: Launch Docker from Applications folder
- **All**: Wait for Docker whale icon in system tray/menu bar

#### 2. Permission Errors
**Error**: `Permission denied` or `Access denied`

**Solutions**:
##### Windows
```cmd
REM Use correct volume syntax for Command Prompt
docker run --rm -v "%cd%:/app" diabetes-pipeline:latest

REM For PowerShell
docker run --rm -v "${PWD}:/app" diabetes-pipeline:latest
```

##### macOS
```bash
# Give Docker access to folders in System Preferences > Security & Privacy
# Use correct path syntax
docker run --rm -v "$(pwd):/app" diabetes-pipeline:latest
```

#### 3. Memory/Performance Issues
**Error**: Container stops unexpectedly or runs slowly

**Solutions**:
- **Windows**: Docker Desktop > Settings > Resources > Increase Memory to 4GB+
- **macOS**: Docker Desktop > Preferences > Resources > Memory > 4GB+
- **All**: Close other applications to free up memory

#### 4. Path Issues
**Error**: `No such file or directory`

**Solutions**:
##### Windows
```cmd
REM Ensure you're in the project directory
cd "C:\Users\ludov\Desktop\Wild_Boy\Magistrale\primo anno\MOTT\Progetto_mott"
dir
REM Should see: Dockerfile, docker-compose.yml, metodi folder, etc.
```

##### macOS/Linux
```bash
# Ensure you're in the project directory
pwd
ls
# Should see: Dockerfile, docker-compose.yml, metodi folder, etc.
```

#### 5. WSL 2 Issues (Windows Only)
**Error**: WSL 2 related errors

**Solutions**:
1. Install WSL 2: `wsl --install`
2. Update WSL: `wsl --update`
3. Set WSL 2 as default: `wsl --set-default-version 2`
4. Restart Docker Desktop

#### 6. Build Errors
**Error**: Docker build fails

**Solutions**:
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build --no-cache -t diabetes-pipeline:latest .
```

## Quick Start Guide (TL;DR)

### For Beginners - Fastest Way to Run

#### Windows Users
1. **Install Docker Desktop**: Download from https://docs.docker.com/desktop/install/windows-install/
2. **Open Command Prompt as Administrator** and navigate to project:
```cmd
cd "C:\Users\ludov\Desktop\Wild_Boy\Magistrale\primo anno\MOTT\Progetto_mott"
```
3. **Build and run**:
```cmd
docker build -t diabetes-pipeline .
docker-compose up
```

#### Mac Users
1. **Install Docker Desktop**: Download from https://docs.docker.com/desktop/install/mac-install/
2. **Open Terminal** and navigate to project:
```bash
cd /path/to/your/Progetto_mott
```
3. **Build and run**:
```bash
docker build -t diabetes-pipeline .
docker-compose up
```

### Expected Output
When successful, you should see:
```
SIMPLE TEST SUITE - DIABETES READMISSION PREDICTION
============================================================
--- Import Clustering ---
PASS: Import clustering system
--- Import Integrato ---
PASS: Import sistema integrato
...
RISULTATO: SISTEMA FUNZIONALE
```

## Maintenance and Cleanup Commands

### Regular Maintenance
```bash
# Stop all containers
docker-compose down

# Remove old unused containers and images (saves disk space)
docker system prune -f

# Check disk usage
docker system df
```

### Complete Cleanup (Nuclear Option)
```bash
# WARNING: This removes everything Docker related
docker-compose down -v
docker system prune -a -f
docker volume prune -f
```

## Support and Help

### Getting Help
```bash
# Docker help
docker --help
docker-compose --help

# Container logs for debugging
docker logs <container-name>
docker-compose logs

# Check running containers
docker ps
```

### Project-Specific Health Checks
```bash
# Quick system health check
docker run --rm diabetes-pipeline:latest python -c "
import sys
try:
    import pandas, numpy, sklearn
    print('✅ All dependencies OK')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"
```