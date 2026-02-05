# 🔧 Fix Vercel Frontend Deployment

## Problem Identified

The Vercel URL **https://frontend-mauve-seven-92.vercel.app/** is serving the wrong app (ModelLab instead of Mayo Clinic STRIP AI).

### Two Vercel Projects:
1. **"mayo-clinic-strip-ai"** - Correct project, but "No Production Deployment"
2. **"frontend"** - Has deployment but connected to wrong repo (ModelLab)

## Solution: Deploy mayo-clinic-strip-ai Project

### Option 1: Deploy via Vercel Dashboard (EASIEST)

1. **Go to Vercel Dashboard**
   - Visit: https://vercel.com/calebs-projects-a6310ab2

2. **Click on "mayo-clinic-strip-ai" project**
   - NOT the "frontend" project

3. **Configure Settings** (if needed):
   - Go to **Settings** → **General**
   - Set **Root Directory** to: `frontend`
   - Framework Preset: Next.js
   - Build Command: (leave default or `npm run build`)
   - Output Directory: (leave default)

4. **Connect to GitHub** (if not connected):
   - Settings → **Git**
   - Connect to: `calebnewtonusc/mayo-clinic-strip-ai`
   - Production Branch: `main`

5. **Set Environment Variables**:
   - Go to **Settings** → **Environment Variables**
   - Add:
     ```
     NEXT_PUBLIC_API_URL=https://mayo-clinic-strip-ai-production.up.railway.app
     NEXT_PUBLIC_API_KEY=cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw
     ```
   - Apply to: **Production**, **Preview**, **Development**

6. **Trigger Deployment**:
   - Go to **Deployments** tab
   - Click **"Deploy"** or **"Redeploy"** button
   - OR: Make any commit and push to GitHub (will auto-deploy)

### Option 2: Fix Team Permissions for CLI Deploy

The CLI error shows:
```
Git author joelnewto@usc.edu must have access to the team Caleb's projects
```

To fix:
1. Go to Vercel Dashboard → **Settings** → **Team**
2. Click **"Members"**
3. Add `joelnewto@usc.edu` as a team member
4. Set role to **Developer** or **Owner**
5. Accept invitation
6. Then CLI deploy will work:
   ```bash
   cd frontend
   vercel --prod
   ```

### Option 3: Disconnect Old "frontend" Project

To clean up:
1. Go to **"frontend"** project (the ModelLab one)
2. Settings → **General**
3. Scroll to bottom → **"Delete Project"**
4. This won't affect your new mayo-clinic-strip-ai deployment

## After Deployment Completes

### Find Your New URL

1. Go to **mayo-clinic-strip-ai** project in Vercel
2. Click on latest deployment
3. Copy the production URL (will be something like):
   - `https://mayo-clinic-strip-ai.vercel.app` OR
   - `https://mayo-clinic-strip-ai-<hash>.vercel.app`

### Update Railway CORS

Once you have the new Vercel URL, update backend CORS:

1. Edit `deploy/api_with_metrics.py` lines 42-45:
   ```python
   CORS(app, resources={
       r"/*": {
           "origins": [
               "https://mayo-clinic-strip-ai.vercel.app",  # New URL
               "http://localhost:3000"
           ]
       }
   })
   ```

2. Commit and push:
   ```bash
   git add deploy/api_with_metrics.py
   git commit -m "Update CORS for new Vercel URL"
   git push origin main
   ```

3. Railway will auto-redeploy with new CORS settings

## Testing

Once both are deployed:

1. **Test Frontend**:
   - Visit your new Vercel URL
   - Should show "Mayo Clinic STRIP AI" title
   - Upload button should be visible

2. **Test Backend**:
   ```bash
   curl https://mayo-clinic-strip-ai-production.up.railway.app/health
   ```
   Should return: `"model_loaded": true`

3. **Test End-to-End**:
   - Upload a medical image via frontend
   - Should get classification results
   - No CORS errors in browser console

## Current Status

- ✅ `.dockerignore` fixed - models/ no longer excluded
- ✅ Latest commit: `7a8cce7` - Fix dockerignore for model bundling
- ✅ Railway should be building now with model included
- ✅ Frontend code has Railway URL hardcoded
- ⏳ Need to deploy mayo-clinic-strip-ai Vercel project
- ⏳ Wait for Railway build to complete

## Next Steps

1. **NOW**: Deploy mayo-clinic-strip-ai Vercel project (via dashboard)
2. **Wait 2-5 min**: Railway build completes
3. **Test**: Check if model loaded on Railway
4. **Test**: Upload image via new frontend URL
5. **Celebrate**: System is fully deployed! 🎉

---

**IMPORTANT**: Don't use the "frontend" project anymore - that's connected to ModelLab. Use "mayo-clinic-strip-ai" project going forward!
