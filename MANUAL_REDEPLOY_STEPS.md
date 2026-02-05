# 🚂 Manual Railway Redeploy Instructions

## Why You Need to Do This

Railway is stuck on deployment `0aa25c14` from 9:38 AM. I pushed commit `8b39a47` to trigger a new build, but if Railway's GitHub webhook isn't working, you need to manually trigger it.

## Steps to Manually Redeploy

### 1. Go to Railway Dashboard
https://railway.com/project/c24f768c-1481-4df5-b8b5-b1b42b9ad2d2

### 2. Find Your Service
- Look for the service card labeled **"mayo-clinic-strip-ai"**
- Click on it

### 3. Go to Deployments Tab
- At the top, click **"Deployments"**
- You should see the current deployment: `0aa25c14` - Active (OLD)

### 4. Trigger New Deployment

**Option A: Deploy Latest Commit**
1. Look for **"Deploy"** button (usually top-right or in the deployments list)
2. Click it
3. Select **"Deploy Latest Commit"** or **"Redeploy"**
4. Confirm

**Option B: Manual Deploy**
1. Click the **"..."** menu (three dots) next to the current deployment
2. Select **"Redeploy"**
3. Confirm

**Option C: Settings Method**
1. Go to **Settings** tab
2. Scroll to **"Service Settings"**
3. Find **"Deploy"** section
4. Click **"Redeploy"** or **"Deploy Latest Commit"**

### 5. Verify New Deployment Started
After triggering, you should see:
- New deployment appear in the list
- Status: **"BUILDING"**
- Latest commit hash: **`8b39a47`** (or similar recent commit)

### 6. Watch Build Progress
1. Click on the new deployment
2. Go to **"Build Logs"** tab
3. Watch for these key steps:

```
Building...
Step 1/12 : FROM python:3.9-slim
...
Step 10/12 : COPY models/ ./models/
 ---> Running in [container-id]
 ---> [hash]
Step 11/12 : EXPOSE 5000
...
Successfully built [image-id]
```

The critical line is: **`Step X/Y : COPY models/ ./models/`**

This confirms the model file is being copied into the Docker image!

## Expected Timeline

- **0-2 min**: Build starts, installs system dependencies
- **2-7 min**: Installs PyTorch and ML libraries (SLOW - biggest step)
- **7-9 min**: Copies code and model file
- **9-10 min**: Container starts, model loads
- **10+ min**: Deployment goes live!

## How to Monitor

### Check Build Status
Railway Dashboard → Deployments → Click new deployment → Build Logs

### Check Runtime Logs
Railway Dashboard → Deployments → Click new deployment → Deploy Logs

Look for:
```
Attempting to load model from /app/models/best_model.pth...
Model loaded successfully: resnet18
Device: cpu
```

### Test API Health
After deployment shows **"ACTIVE"**, test:
```bash
curl https://mayo-clinic-strip-ai-production.up.railway.app/health
```

Should return:
```json
{
  "device": "cpu",
  "model_loaded": true,  ← Should be TRUE now!
  "status": "healthy",
  "debug": {
    "model_checkpoint_env": "/app/models/best_model.pth",
    "model_file_exists": true,
    "models_dir_exists": true,
    "models_dir_contents": ["best_model.pth"]
  },
  "model_info": {
    "architecture": "resnet18",
    "num_classes": 2
  }
}
```

## If Build Fails

### Check Build Logs for Errors

**Common errors:**

1. **Model file not found**:
```
COPY failed: file not found in build context
```
Solution: Verify models/best_model.pth exists in GitHub repo

2. **Out of memory**:
```
Error: Process out of memory
```
Solution: Upgrade Railway plan or reduce model size

3. **Dependency conflict**:
```
ERROR: Cannot install ...
```
Solution: Check requirements.txt for conflicts

### Check Environment Variables

Go to: Railway Dashboard → Service → Variables

Ensure these are set:
```
MODEL_CHECKPOINT=/app/models/best_model.pth
FLASK_ENV=production
API_HOST=0.0.0.0
API_PORT=5000
API_KEY=cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw
```

## After Successful Deployment

### Test Backend
```bash
# Health check (should show model_loaded: true)
curl https://mayo-clinic-strip-ai-production.up.railway.app/health

# Model info
curl https://mayo-clinic-strip-ai-production.up.railway.app/model_info
```

### Test Frontend
1. Go to: https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app
2. Upload a medical image
3. Click "Classify Blood Clot"
4. Should see prediction results!

## Troubleshooting

### Railway Not Detecting GitHub Push

If Railway webhook isn't working:

1. **Check Repository Settings**:
   - Railway Dashboard → Settings → "Connected Repo"
   - Verify it shows: `calebnewtonusc/mayo-clinic-strip-ai`

2. **Reconnect Repository**:
   - Settings → Disconnect Repo
   - Reconnect to GitHub
   - Redeploy

3. **Check Branch**:
   - Ensure Railway is deploying from **`main`** branch
   - Settings → "Deploy Branch" should be `main`

### Still Not Working?

Contact me with:
1. Screenshot of Railway deployments page
2. Build logs (if any)
3. Runtime logs (if deployment succeeded)
4. Error messages

---

## Summary

**What I did**:
- ✅ Pushed commit `8b39a47` to trigger GitHub webhook
- ✅ This commit includes all fixes:
  - Bundled model file in Docker image
  - Debug endpoint to diagnose issues
  - Fixed frontend API URL
  - All previous bug fixes

**What you need to do**:
- 🔵 **Manually trigger redeploy** in Railway dashboard (follow steps above)
- 🔵 **Watch build logs** to ensure model file is copied
- 🔵 **Test health endpoint** after deployment completes
- 🔵 **Test frontend** to verify predictions work

**Expected result**:
- Model loads successfully on startup
- Health check shows `model_loaded: true`
- Frontend can classify images
- Full end-to-end system works!

---

**Do it now!** Go to the Railway dashboard and click that deploy button! 🚀
