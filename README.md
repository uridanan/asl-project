# asl-project

This app can recognize ASL fingerspelling and translate hand gestures into text

Install
Step 1: clone the repo

git clone https://github.com/lielledanan/asl-project

Step 2: install requirements

pip install -r requirements.txt

Step 3: run the streamlit app

streamlit run app.py

---

## Run with Docker (Local)

To run the application locally inside a Docker container:

1. **Build the Docker Image**:
   ```bash
   docker build -t asl-translator .
   ```

2. **Run the Container**:
   ```bash
   docker run -d -p 8501:8501 --name asl-app asl-translator
   ```

3. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

---

## Deploy to GCP Cloud Run via Cloud Build

The project includes configuration for automated builds and deployment to Google Cloud Run in the `me-west1` region under the project `playground-488120`.

### Prerequisites
1. **Google Cloud SDK** installed and configured (`gcloud auth login`).
2. **Artifact Registry** repository named `asl-project` created in region `me-west1` under project `playground-488120`:
   ```bash
   gcloud artifacts repositories create asl-project \
       --repository-format=docker \
       --location=me-west1 \
       --description="Docker repository for ASL Translator"
   ```
3. **Cloud Build Service Account Permissions**:
   Ensure the Cloud Build service account has permission to deploy to Cloud Run and access/write resources.

### Run Build & Deploy Command
You can trigger the build and deployment process directly from your terminal using:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

This will:
1. Pack and upload the project code.
2. Build the Docker image in GCP Cloud Build.
3. Push the image to GCP Artifact Registry (`me-west1-docker.pkg.dev/playground-488120/asl-project/asl-translator`).
4. Deploy a new revision to Google Cloud Run (`asl-translator`) with public unauthenticated access allowed.

