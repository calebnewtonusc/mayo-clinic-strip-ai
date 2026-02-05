# 🚀 Complete Deployment Guide - Mayo Clinic STRIP AI

Deploy your ML medical imaging app to production in ~30 minutes!

**Architecture:**
- **Backend:** Railway.app (Flask API + PyTorch models)
- **Frontend:** Vercel (React/Next.js web interface)
- **Source Control:** GitHub

---

## 📋 Prerequisites

1. **Accounts (all free tier available):**
   - [ ] GitHub account
   - [ ] Railway.app account (free $5 credit/month)
   - [ ] Vercel account (free unlimited deploys)

2. **Local Tools:**
   - [ ] Git installed
   - [ ] Trained model file (`best_model.pth`)

---

## Part 1: Setup GitHub Repository (5 minutes)

### Step 1: Initialize Git & Push to GitHub

```bash
cd /Users/joelnewton/Desktop/2026-Code/projects/mayo-clinic-strip-ai

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Mayo Clinic STRIP AI production-ready"

# Create GitHub repo (via GitHub website or CLI)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/mayo-clinic-strip-ai.git
git branch -M main
git push -u origin main
```

**⚠️ IMPORTANT:** Your trained model file (`*.pth`) is in `.gitignore` - you'll upload it separately to Railway.

---

## Part 2: Deploy Backend to Railway (10 minutes)

### Step 1: Sign Up & Create Project

1. Go to [railway.app](https://railway.app) and sign up
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Connect your GitHub account
4. Select `mayo-clinic-strip-ai` repository

### Step 2: Configure Railway

Railway will auto-detect the Dockerfile. Configure these settings:

#### A. Environment Variables

In Railway dashboard → your service → **Variables** tab, add:

```
FLASK_ENV=production
API_HOST=0.0.0.0
API_PORT=5000
MODEL_CHECKPOINT=/app/models/best_model.pth
API_KEY=your-secret-api-key-here-change-this
```

**Generate secure API_KEY:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### B. Upload Model File

Railway doesn't support large files in git. Upload your model:

**Option 1: Use Railway Volume (Recommended)**
1. In Railway → **Settings** → **Volumes**
2. Add volume: `/app/models`
3. After first deploy, use Railway CLI to upload:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Upload model file
railway run scp best_model.pth /app/models/best_model.pth
```

**Option 2: Use Cloud Storage**
- Upload model to S3/GCS/Azure Blob
- Download in Docker container on startup
- Add download script to Dockerfile

### Step 3: Deploy

1. Railway auto-deploys on git push
2. Check **Deployments** tab for build logs
3. Once deployed, Railway gives you a URL: `https://your-app.up.railway.app`

**Test your API:**
```bash
curl https://your-app.up.railway.app/health
# Should return: {"status": "healthy", ...}
```

---

## Part 3: Deploy Frontend to Vercel (10 minutes)

### Step 1: Sign Up & Import Project

1. Go to [vercel.com](https://vercel.com) and sign up with GitHub
2. Click **"Add New..."** → **"Project"**
3. Import `mayo-clinic-strip-ai` repository
4. **Important:** Set **Root Directory** to `frontend/`

### Step 2: Configure Build Settings

Vercel should auto-detect Next.js. Verify:

- **Framework Preset:** Next.js
- **Build Command:** `npm run build`
- **Output Directory:** `.next`
- **Install Command:** `npm install`

### Step 3: Set Environment Variables

In Vercel → **Settings** → **Environment Variables**, add:

```
NEXT_PUBLIC_API_URL=https://your-app.up.railway.app
```

Replace `your-app.up.railway.app` with your actual Railway URL from Part 2.

### Step 4: Deploy

1. Click **"Deploy"**
2. Vercel builds and deploys (2-3 minutes)
3. Your app is live at: `https://your-project.vercel.app`

---

## Part 4: Configure CORS (Important!)

Your backend needs to allow requests from your Vercel frontend.

### Update deploy/api_with_metrics.py

Add after `CORS(app)`:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://your-project.vercel.app",
            "http://localhost:3000"  # For local development
        ]
    }
})
```

Commit and push:
```bash
git add deploy/api_with_metrics.py
git commit -m "Configure CORS for Vercel frontend"
git push
```

Railway auto-redeploys.

---

## Part 5: Test Your Deployment (5 minutes)

### Frontend Test

1. Visit your Vercel URL: `https://your-project.vercel.app`
2. Upload a test image
3. Click "Classify Blood Clot"
4. Should see prediction results

### API Test

```bash
# Health check
curl https://your-app.up.railway.app/health

# Test prediction
curl -X POST https://your-app.up.railway.app/predict \
  -H "X-API-Key: your-secret-api-key" \
  -F "file=@test_image.png"
```

---

## 🔧 Local Development

### Run Backend Locally

```bash
# Terminal 1: Backend
export MODEL_CHECKPOINT=experiments/baseline_experiment/checkpoints/best_model.pth
export API_KEY=dev-key
python deploy/api_with_metrics.py
# Runs on http://localhost:5000
```

### Run Frontend Locally

```bash
# Terminal 2: Frontend
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:5000" > .env.local
npm run dev
# Runs on http://localhost:3000
```

---

## 📊 Monitoring & Logs

### Railway Logs
- Dashboard → your service → **Logs** tab
- Real-time logs from your API

### Vercel Logs
- Dashboard → your project → **Deployments**
- Click deployment → **Build Logs** or **Function Logs**

---

## 💰 Cost Estimate

| Service | Free Tier | Paid (if needed) |
|---------|-----------|------------------|
| **Railway** | $5 credit/month (~100 hours) | $0.000231/GB-hour (~$10-20/month) |
| **Vercel** | Unlimited deploys | $20/month Pro (if you need more) |
| **GitHub** | Unlimited public repos | $4/month private (optional) |

**Total:** Free for development/demo, ~$10-30/month for production

---

## 🐛 Troubleshooting

### "Model file not found"
- Check Railway volumes are mounted correctly
- Verify `MODEL_CHECKPOINT` env var path
- Check deployment logs for errors

### "CORS error" in frontend
- Add your Vercel domain to CORS origins
- Check browser console for exact error
- Verify `NEXT_PUBLIC_API_URL` is correct

### "API returns 401 Unauthorized"
- Set `X-API-Key` header in requests
- Check API_KEY env var in Railway
- Frontend: API key should NOT be in frontend (security risk)

### Railway build fails
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Check Railway build logs for specific errors

### Vercel build fails
- Verify Root Directory is set to `frontend/`
- Check Node.js version compatibility
- Review build logs in Vercel dashboard

---

## 🚀 Next Steps

### Production Checklist

- [ ] Set strong API_KEY (use secrets.token_urlsafe(32))
- [ ] Configure custom domain (Vercel + Railway support this)
- [ ] Set up monitoring (Railway metrics, Sentry for errors)
- [ ] Enable HTTPS only (both platforms do this by default)
- [ ] Add rate limiting to API
- [ ] Set up database for logging predictions (optional)
- [ ] Configure backup for model files
- [ ] Add authentication to frontend (NextAuth.js)
- [ ] Set up CI/CD for tests before deploy
- [ ] Add model versioning

### Scaling

**When you need more power:**

1. **Railway:** Upgrade to hobby plan ($5-20/month)
   - More RAM/CPU for model inference
   - Persistent storage

2. **Alternative ML Platforms:**
   - Modal.com (serverless ML, pay per use)
   - Hugging Face Spaces (free GPU!)
   - AWS Lambda with containers (complex but scalable)

---

## 📚 Useful Commands

```bash
# Update frontend
cd frontend && npm run build && git add . && git commit -m "Update frontend" && git push

# Update backend
git add deploy/ src/ && git commit -m "Update API" && git push

# View Railway logs
railway logs --tail

# Redeploy Vercel
vercel --prod
```

---

## ✅ Success Checklist

- [x] GitHub repo created and pushed
- [x] Railway backend deployed and API working
- [x] Model file uploaded to Railway
- [x] Vercel frontend deployed
- [x] CORS configured correctly
- [x] Can upload image and get prediction
- [x] API key authentication working
- [x] Health check returning success

---

**Congratulations! Your ML medical imaging app is now deployed to production!** 🎉

**Your URLs:**
- **Frontend:** https://your-project.vercel.app
- **Backend API:** https://your-app.up.railway.app
- **Health Check:** https://your-app.up.railway.app/health

**Questions?** Check the troubleshooting section or Railway/Vercel docs.

---

**Last Updated:** February 5, 2026
**Status:** ✅ Complete Deployment Guide
