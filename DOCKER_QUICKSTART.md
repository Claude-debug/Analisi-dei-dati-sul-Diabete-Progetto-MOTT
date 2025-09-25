# Docker Quick Start - Diabetes Prediction Pipeline

## For Complete Beginners

### 30-Second Setup

#### Windows
1. **Download Docker**: https://docs.docker.com/desktop/install/windows-install/
2. **Install & Start Docker Desktop**
3. **Open Command Prompt as Administrator**:
```cmd
cd "C:\Users\ludov\Desktop\Wild_Boy\Magistrale\primo anno\MOTT\Progetto_mott"
docker build -t diabetes-pipeline .
docker-compose up
```

#### macOS
1. **Download Docker**: https://docs.docker.com/desktop/install/mac-install/
2. **Install & Start Docker Desktop**
3. **Open Terminal**:
```bash
cd /path/to/your/Progetto_mott
docker build -t diabetes-pipeline .
docker-compose up
```

### Success Indicators
You'll see:
```
SIMPLE TEST SUITE - DIABETES READMISSION PREDICTION
--- Import Clustering ---
PASS: Import clustering system
RISULTATO: SISTEMA FUNZIONALE
```

### Common Problems & Quick Fixes

**"Docker not running"** → Start Docker Desktop application
**"Permission denied"** → Run as Administrator (Windows) or check Docker folder permissions (Mac)
**"Build fails"** → Ensure you're in the correct project folder

### What Each Service Does

- **diabetes-pipeline**: Runs basic system tests
- **clustering-analysis**: Compares 4 clustering methods
- **integrated-system**: Runs final prediction system
- **full-test-suite**: Complete system validation

### Stop Everything
```bash
docker-compose down
```

For detailed instructions, see `docker-commands.md`