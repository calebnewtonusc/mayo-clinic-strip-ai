#!/bin/bash
# Automated Railway Deployment Script
set -e

echo "🚀 Mayo Clinic STRIP AI - Railway Auto-Deploy"
echo "=============================================="

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "⚠️  Not logged into Railway. Opening browser for authentication..."
    railway login
    echo "✅ Authentication complete!"
fi

echo ""
echo "📦 Creating Railway project..."
railway init --name mayo-clinic-strip-ai

echo ""
echo "🔗 Linking to GitHub repository..."
railway link

echo ""
echo "⚙️  Setting environment variables..."
railway variables set FLASK_ENV=production
railway variables set API_HOST=0.0.0.0
railway variables set API_PORT=5000
railway variables set MODEL_CHECKPOINT=/app/models/best_model.pth

# Generate secure API key
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
railway variables set API_KEY=$API_KEY

echo ""
echo "🔐 Generated API Key: $API_KEY"
echo "   (Save this for frontend configuration!)"

echo ""
echo "📁 Creating volume for model files..."
railway volume create models --mount /app/models

echo ""
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "⏳ Waiting for deployment to complete..."
railway status --watch

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
echo "📊 Your Railway deployment:"
railway status

echo ""
echo "🌐 Getting your Railway URL..."
RAILWAY_URL=$(railway domain)
echo "   Backend URL: $RAILWAY_URL"

echo ""
echo "=============================================="
echo "✅ SUCCESS! Next steps:"
echo "=============================================="
echo "1. Upload your model file:"
echo "   railway run cp /path/to/best_model.pth /app/models/"
echo ""
echo "2. Update Vercel frontend with Railway URL:"
echo "   vercel env add NEXT_PUBLIC_API_URL production"
echo "   Then paste: $RAILWAY_URL"
echo ""
echo "3. Update CORS in deploy/api_with_metrics.py:"
echo "   Add: https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app"
echo ""
echo "4. Test your deployment:"
echo "   curl $RAILWAY_URL/health"
echo "=============================================="
