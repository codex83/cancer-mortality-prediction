# Quick Start Guide

## 🚀 3-Minute GitHub Upload

### Step 1: Open Terminal & Navigate
```bash
cd /Users/hritikjhaveri/Desktop/UChicago/Q4/ML_Ops/Assignments/Assignment4/cancer-mortality-prediction
```

### Step 2: Initialize Git
```bash
git init
git add .
git commit -m "Initial commit: Cancer Mortality Prediction with ML monitoring"
```

### Step 3: Create Repository on GitHub
1. Go to: https://github.com/new
2. Name: `cancer-mortality-prediction`
3. Description: "ML pipeline for predicting cancer mortality rates with Evidently AI monitoring"
4. Public repository
5. Click "Create repository"

### Step 4: Push to GitHub
**Replace `YOUR-USERNAME` with your GitHub username:**
```bash
git remote add origin https://github.com/YOUR-USERNAME/cancer-mortality-prediction.git
git branch -M main
git push -u origin main
```

### Step 5: Update Your Info
1. Edit `README.md` - Update lines with your:
   - GitHub username
   - LinkedIn profile
   - Email address

2. Save and push:
```bash
git add README.md
git commit -m "Update personal information"
git push
```

---

## ✅ Done!

Your project is now live on GitHub! 🎉

Visit: `https://github.com/YOUR-USERNAME/cancer-mortality-prediction`

---

## 📝 Next Actions

### On GitHub (5 minutes)
- Click ⚙️ next to "About" → Add topics: `machine-learning`, `python`, `scikit-learn`, `healthcare`
- Click "Releases" → "Create a new release" → Tag: `v1.0.0`
- Enable "Issues" and "Discussions" in Settings

### Share Your Work (10 minutes)
- Update your resume/CV
- Add to LinkedIn portfolio
- Share on social media
- Add link to your personal website

---

## 🔧 Common Issues

### "Permission denied (publickey)"
Use HTTPS instead:
```bash
git remote set-url origin https://github.com/YOUR-USERNAME/cancer-mortality-prediction.git
```

### "Repository not found"
Double-check:
1. Repository name is correct
2. Username is correct
3. Repository exists on GitHub

### "Failed to push some refs"
Pull first, then push:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 📚 Full Documentation

- **Detailed Setup**: See `GITHUB_SETUP.md`
- **Project Info**: See `PROJECT_SUMMARY.md`
- **Usage Guide**: See `README.md`

---

**Need Help?** Open an issue on GitHub or refer to the documentation files.

Good luck! 🌟

