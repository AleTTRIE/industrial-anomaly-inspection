# Deploying to Google Cloud Run

Prerequisites: a GCP project, `gcloud` installed and authenticated, and a **budget alert** set before anything else (Billing > Budgets & alerts).

```bash
PROJECT_ID=<your-project>
REGION=europe-west1
REPO=inspection
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/api:latest

gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION

gcloud auth configure-docker $REGION-docker.pkg.dev
docker build -t $IMAGE .
docker push $IMAGE

gcloud run deploy inspection-api \
  --image $IMAGE \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 1
```

`--min-instances 0` scales to zero when idle (no cost, but the first request is slow: cold start). `--max-instances 1` caps the bill if the public URL gets hammered.

## TODO: continuous deployment

Deploy from GitHub Actions on version tags (`v*`), authenticating with Workload Identity Federation rather than a service-account JSON key stored in GitHub secrets.
