# 🚀 Railway Pro Deployment - Active Now!

## ✅ What Just Happened:

1. **Railway Pro Plan Activated** - Full PyTorch support!
2. **Code Pushed to GitHub** - Triggered automatic Railway rebuild
3. **Build In Progress** - Railway is deploying your app now

## 📊 Monitor Your Deployment:

**Dashboard**: https://railway.com/project/c24f768c-1481-4df5-b8b5-b1b42b9ad2d2

Watch the **Build Logs** to see progress. With Pro plan, the full PyTorch build should complete successfully.

## 🔧 Next Steps (Once Build Completes):

### 1. Add Environment Variables
In Railway dashboard → Your service → **Variables** tab:

```env
FLASK_ENV=production
API_HOST=0.0.0.0
API_PORT=5000
MODEL_CHECKPOINT=/app/models/best_model.pth
API_KEY=cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw
```

### 2. Generate Public Domain
- **Settings** tab → **Networking** → **Generate Domain**
- Copy the URL (e.g., `mayo-clinic-strip-ai-production.up.railway.app`)

### 3. Upload Your Model File
Option A - Via Railway CLI:
```bash
railway run cp experiments/baseline_experiment/checkpoints/best_model.pth /app/models/best_model.pth
```

Option B - Via Railway Volumes (in dashboard):
- **Settings** → **Volumes** → Create volume at `/app/models`
- Upload `best_model.pth` through Railway CLI

### 4. Update Vercel Frontend
Once you have your Railway URL:
```bash
cd /Users/joelnewton/Desktop/2026-Code/projects/mayo-clinic-strip-ai/frontend
vercel env add NEXT_PUBLIC_API_URL production
# Paste your Railway URL when prompted
vercel --prod
```

### 5. Configure CORS
Update [deploy/api_with_metrics.py](deploy/api_with_metrics.py:172):

```python
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app",
            "http://localhost:3000"
        ]
    }
})
```

Then:
```bash
git add deploy/api_with_metrics.py
git commit -m "Add CORS for Vercel frontend"
git push
```

## 🔑 Important Info:

- **API Key**: `cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw` (keep secret!)
- **Vercel Frontend**: https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app
- **GitHub Repo**: https://github.com/calebnewtonusc/mayo-clinic-strip-ai

## 🧪 Test After Deployment:

```bash
# Health check (replace with your Railway URL)
curl https://your-app.up.railway.app/health

# Test prediction
curl -X POST https://your-app.up.railway.app/predict \
  -H "X-API-Key: cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw" \
  -F "file=@test_image.png"
```

---

**Current Status**: Build in progress with Railway Pro plan ✨

Check your dashboard to monitor the deployment!
