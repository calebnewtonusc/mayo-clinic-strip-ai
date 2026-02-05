#!/bin/bash

# Auto-upload model script
# Polls the Railway deployment and uploads the model when ready

API_URL="https://mayo-clinic-strip-ai-production.up.railway.app"
API_KEY="cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw"
MODEL_FILE="/Users/joelnewton/Desktop/2026-Code/projects/mayo-clinic-strip-ai/experiments/checkpoints/best_model.pth"

echo "==================================="
echo "Auto Model Upload Script"
echo "==================================="
echo "Waiting for Railway deployment to complete..."
echo "This usually takes 5-10 minutes for PyTorch apps."
echo ""

MAX_ATTEMPTS=40  # 40 attempts * 30 seconds = 20 minutes max
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[Attempt $ATTEMPT/$MAX_ATTEMPTS] Checking if /upload-model endpoint is ready..."

    # Test if endpoint exists (will return 400 "No file provided" if ready, 404 if not deployed)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        -H "X-API-Key: $API_KEY" \
        "$API_URL/upload-model")

    if [ "$HTTP_CODE" == "400" ] || [ "$HTTP_CODE" == "200" ]; then
        echo ""
        echo "✅ Endpoint is ready! Uploading model..."
        echo ""

        # Upload the model
        RESPONSE=$(curl -X POST \
            -H "X-API-Key: $API_KEY" \
            -F "file=@$MODEL_FILE" \
            "$API_URL/upload-model" \
            -w "\nHTTP_CODE:%{http_code}" 2>&1)

        UPLOAD_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)

        if [ "$UPLOAD_CODE" == "200" ]; then
            echo "🎉 SUCCESS! Model uploaded and loaded!"
            echo ""
            echo "$RESPONSE" | grep -v "HTTP_CODE:"
            echo ""
            echo "Your API is now ready to make predictions!"
            echo ""
            echo "Frontend: https://frontend-aus71qall-calebs-projects-a6310ab2.vercel.app"
            echo "Backend: $API_URL"
            echo ""
            exit 0
        else
            echo "❌ Upload failed with status code: $UPLOAD_CODE"
            echo "$RESPONSE" | grep -v "HTTP_CODE:"
            exit 1
        fi
    fi

    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        echo "   Endpoint not ready yet (HTTP $HTTP_CODE). Waiting 30 seconds..."
        sleep 30
    fi
done

echo ""
echo "❌ Timeout: Deployment did not complete within 20 minutes."
echo "Please check Railway dashboard: https://railway.com/project/c24f768c-1481-4df5-b8b5-b1b42b9ad2d2"
exit 1
