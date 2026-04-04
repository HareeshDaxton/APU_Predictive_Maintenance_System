#!/bin/bash
# ============================================================
#  APU Predictive Maintenance — Cloud Scheduler Setup
#  Run this script ONCE after the retrain Docker image is pushed.
#
#  BEFORE RUNNING:
#    1. Replace YOUR_PROJECT_ID with your actual GCP project ID
#    2. Make sure the apu-retrain image is pushed to Artifact Registry
#    3. Run from PowerShell:  bash infra/setup_scheduler.sh
# ============================================================

PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/apu-repo/apu-retrain:latest"
SA_EMAIL="apu-deploy-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=============================================="
echo " APU — Cloud Scheduler Setup"
echo " Project  : $PROJECT_ID"
echo " Region   : $REGION"
echo "=============================================="

# ── Step 1: Create Cloud Run Job for retraining ──────────────
echo ""
echo "[1/3] Creating Cloud Run Job: apu-retrain-job ..."
gcloud run jobs create apu-retrain-job \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --memory 2Gi \
  --cpu 2 \
  --task-timeout 3600 \
  --set-env-vars "GCS_MODEL_BUCKET=apu-model-artifacts-${PROJECT_ID},GCS_DATA_BUCKET=apu-incoming-data-${PROJECT_ID},GCP_PROJECT_ID=${PROJECT_ID}"

echo "Cloud Run Job created."

# ── Step 2: Test-run the job ONCE to verify it works ─────────
echo ""
echo "[2/3] Running a test execution of the job (wait ~2 min)..."
gcloud run jobs execute apu-retrain-job \
  --region "$REGION" \
  --wait
echo "Test run complete. Check output above for errors."

# ── Step 3: Create Cloud Scheduler to run every Sunday 00:00 UTC ─
echo ""
echo "[3/3] Creating Cloud Scheduler: apu-weekly-retrain ..."
gcloud scheduler jobs create http apu-weekly-retrain \
  --location "$REGION" \
  --schedule "0 0 * * 0" \
  --uri "https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/apu-retrain-job:run" \
  --message-body "{}" \
  --oauth-service-account-email "$SA_EMAIL" \
  --time-zone "UTC"

echo ""
echo "=============================================="
echo " DONE. Cloud Scheduler created."
echo " Retraining runs every Sunday at 00:00 UTC."
echo " To trigger manually:"
echo "   gcloud run jobs execute apu-retrain-job --region $REGION"
echo "=============================================="
