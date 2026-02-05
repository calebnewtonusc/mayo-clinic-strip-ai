# 🚂 Railway Deployment - Final Setup Steps

## ✅ What's Already Done:

1. **Railway Project Created**: https://railway.com/project/c24f768c-1481-4df5-b8b5-b1b42b9ad2d2
2. **Code Uploaded & Building**: Your app is being deployed right now!
3. **Vercel Frontend Live**: https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app

## 🔧 Quick Setup (2 minutes in Railway Dashboard):

### Step 1: Add Environment Variables
Click on your service → **Variables** tab → Add these:

```
FLASK_ENV=production
API_HOST=0.0.0.0
API_PORT=5000
MODEL_CHECKPOINT=/app/models/best_model.pth
API_KEY=cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw
```

### Step 2: Generate Public Domain
1. Go to **Settings** tab
2. Under **Networking** → Click **Generate Domain**
3. Copy the URL (e.g., `https://mayo-clinic-strip-ai-production.up.railway.app`)

### Step 3: Update Vercel Frontend
Run this command with your Railway URL:

```bash
cd /Users/joelnewton/Desktop/2026-Code/projects/mayo-clinic-strip-ai/frontend
vercel env add NEXT_PUBLIC_API_URL production
# Paste your Railway URL when prompted
vercel --prod
```

### Step 4: Update CORS (Important!)
Edit `deploy/api_with_metrics.py` line 172:

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

Then commit and push:
```bash
git add deploy/api_with_metrics.py
git commit -m "Configure CORS for Vercel frontend"
git push
```

Railway will auto-redeploy!

### Step 5: Upload Your Model File
Once deployed, upload your trained model:

```bash
# Using Railway CLI
railway run cp experiments/baseline_experiment/checkpoints/best_model.pth /app/models/best_model.pth
```

**OR** use Railway Volumes in the dashboard:
1. **Settings** → **Volumes** → Create volume at `/app/models`
2. Use Railway CLI to upload file

## 🧪 Test Your Deployment:

```bash
# Replace with your Railway URL
RAILWAY_URL="https://your-app.up.railway.app"

# Health check
curl $RAILWAY_URL/health

# Test prediction (after model upload)
curl -X POST $RAILWAY_URL/predict \
  -H "X-API-Key: cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw" \
  -F "file=@test_image.png"
```

## 📊 Your Deployment URLs:

- **Railway Dashboard**: https://railway.com/project/c24f768c-1481-4df5-b8b5-b1b42b9ad2d2
- **Build Logs**: https://railway.com/project/c24f768c-1481-4df5-b8b5-b1b42b9ad2d2/service/0ea18baa-b7d9-4026-8e12-dc5c82ce560d?id=7402dc64-7db5-4550-a023-f6035646ab7c
- **Vercel Frontend**: https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app
- **Backend API**: (Get from Railway after domain generation)

## 🔑 Important Info:

**API Key (keep secret!)**: `cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw`

---

**Need help?** Check the comprehensive guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
