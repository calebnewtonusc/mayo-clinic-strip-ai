# 🚂 Railway Web UI Setup (2 Minutes)

Your Railway dashboard is open. Here's exactly what to do:

## Step 1: Create New Service from GitHub

1. In Railway dashboard, click **"+ New"** or **"New Service"**

2. Select **"GitHub Repo"**

3. Find and select: **`calebnewtonusc/mayo-clinic-strip-ai`**

4. Railway will auto-detect the Dockerfile and start building!

---

## Step 2: Add Environment Variables

Once the service is created:

1. Click on your new service card

2. Go to **"Variables"** tab

3. Click **"+ Add Variable"** and add these one by one:

```
FLASK_ENV=production
API_HOST=0.0.0.0
API_PORT=5000
MODEL_CHECKPOINT=/app/models/best_model.pth
API_KEY=cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw
```

4. Click **"Save"** or they auto-save

---

## Step 3: Wait for Build

The service will automatically deploy. Watch the **"Deployments"** tab:

- ✅ You'll see it installing PyTorch (takes ~3-5 min with Pro plan)
- ✅ Flask 2.3.3 (fixed dependency!)
- ✅ All other packages

---

## Step 4: Generate Public Domain

Once deployed successfully:

1. Go to **"Settings"** tab

2. Under **"Networking"** section

3. Click **"Generate Domain"**

4. Copy your URL (e.g., `https://mayo-clinic-strip-ai-production.up.railway.app`)

---

## You're Done! 🎉

Your backend will be live at the Railway URL.

**Next**: Come back here and give me your Railway URL so I can:
- Update your Vercel frontend
- Configure CORS
- Test the full deployment

---

**Current Status:**
- ✅ Code pushed to GitHub with fixed Flask dependency
- ✅ Railway Pro plan active
- 🔄 Ready to create service from GitHub repo
