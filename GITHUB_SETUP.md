# GitHub Repository Setup Guide

Follow these steps to upload your project to GitHub.

## Prerequisites

- Git installed on your computer
- GitHub account created
- Project directory ready

## Step 1: Initialize Git Repository

```bash
cd cancer-mortality-prediction

# Initialize git repository
git init

# Add all files to staging
git add .

# Make your first commit
git commit -m "Initial commit: Cancer Mortality Prediction with ML monitoring"
```

## Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon in the top right
3. Select **"New repository"**
4. Configure your repository:
   - **Repository name**: `cancer-mortality-prediction`
   - **Description**: "ML pipeline for predicting cancer mortality rates with Evidently AI monitoring"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 3: Connect Local Repository to GitHub

Replace `yourusername` with your actual GitHub username:

```bash
# Add GitHub remote
git remote add origin https://github.com/yourusername/cancer-mortality-prediction.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Update README with Your Information

Edit `README.md` and update:

1. **GitHub badge URL** (line 4):
   ```markdown
   [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
   ```

2. **Clone URL** (in Installation section):
   ```bash
   git clone https://github.com/YOUR-USERNAME/cancer-mortality-prediction.git
   ```

3. **Contact section** (at the bottom):
   ```markdown
   **Hritik Jhaveri**
   - GitHub: [@YOUR-USERNAME](https://github.com/YOUR-USERNAME)
   - LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR-PROFILE)
   - Email: your.email@example.com
   ```

4. Commit and push the changes:
   ```bash
   git add README.md
   git commit -m "Update README with personal information"
   git push
   ```

## Step 5: Configure Repository Settings

### Add Topics

Go to your repository on GitHub → Click ⚙️ (Settings gear) next to "About" → Add topics:
- `machine-learning`
- `python`
- `scikit-learn`
- `data-science`
- `evidently-ai`
- `model-monitoring`
- `healthcare`
- `gradient-boosting`
- `jupyter-notebook`

### Add Description

In the same "About" section, add:
```
ML pipeline for predicting county-level cancer mortality rates with comprehensive model monitoring and drift detection
```

### Enable Issues and Discussions (Optional)

Settings → General → Features:
- ✅ Issues
- ✅ Discussions (for community Q&A)

## Step 6: Create Releases (Optional)

1. Go to **Releases** → **Create a new release**
2. **Tag**: `v1.0.0`
3. **Title**: `Initial Release - v1.0.0`
4. **Description**:
   ```markdown
   ## 🎉 Initial Release

   First production-ready version of the Cancer Mortality Prediction project.

   ### Features
   - ✅ Complete ML pipeline with Gradient Boosting
   - ✅ Evidently AI monitoring integration
   - ✅ Drift detection across 3 scenarios
   - ✅ Interactive Jupyter notebook
   - ✅ Modular Python implementation
   - ✅ Comprehensive documentation

   ### Installation
   ```bash
   git clone https://github.com/YOUR-USERNAME/cancer-mortality-prediction.git
   cd cancer-mortality-prediction
   pip install -r requirements.txt
   ```

   ### Quick Start
   ```bash
   jupyter notebook cancer_mortality_prediction.ipynb
   ```
   ```
5. Click **Publish release**

## Step 7: Add Repository Social Preview (Optional)

1. Settings → General → Scroll down to "Social preview"
2. Upload an image:
   - Create a simple graphic with project title
   - Or screenshot of your notebook/dashboard
   - Recommended size: 1280x640px

## Common Git Commands

### Checking Status
```bash
git status                    # See changed files
git log --oneline            # View commit history
```

### Making Changes
```bash
git add .                    # Stage all changes
git add filename.py          # Stage specific file
git commit -m "message"      # Commit changes
git push                     # Push to GitHub
```

### Branching
```bash
git checkout -b feature-name  # Create and switch to new branch
git checkout main            # Switch back to main
git merge feature-name       # Merge branch into current
```

### Updating from GitHub
```bash
git pull                     # Pull latest changes
```

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:

**Option 1: Use Personal Access Token**
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` permissions
3. Use token as password when prompted

**Option 2: Use SSH**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key and add to GitHub
cat ~/.ssh/id_ed25519.pub

# Change remote URL to SSH
git remote set-url origin git@github.com:YOUR-USERNAME/cancer-mortality-prediction.git
```

### Large File Issues

If you get errors about file size:
```bash
# Remove large files from tracking
echo "models/*.pkl" >> .gitignore
echo "evidently_reports/*.html" >> .gitignore

# Remove from git history if already committed
git rm --cached models/*.pkl
git rm --cached evidently_reports/*.html
```

## Next Steps

After uploading to GitHub:

1. ⭐ **Star your own repository** to show it's active
2. 📝 **Add a description and topics** for discoverability
3. 📊 **Enable GitHub Insights** to track activity
4. 🔗 **Share on LinkedIn** with project highlights
5. 📧 **Add to your portfolio** or resume
6. 🤝 **Invite collaborators** if working with others

## Making Your Project Stand Out

### Add Badges to README

Visit [shields.io](https://shields.io/) to create custom badges:
- Build status
- Code coverage
- Documentation status
- Downloads count

### Create GitHub Actions (Optional)

Add CI/CD workflow in `.github/workflows/python-app.yml`:
```yaml
name: Python Application

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
```

---

**Congratulations! Your project is now live on GitHub!** 🎉

Don't forget to keep it updated and respond to issues/PRs from the community.

