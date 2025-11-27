# 🚀 Deployment Guide

This guide explains how to deploy your **Chest X-Ray Disease Detection** application to the web.

## Option 1: Render (Easiest & Free Tier)
Render is a modern cloud provider that is very easy to use and has a free tier.

1.  **Push your code to GitHub**: Make sure your project is in a GitHub repository.
2.  **Sign up for Render**: Go to [render.com](https://render.com) and sign up/login.
3.  **Create a Web Service**:
    *   Click "New +" and select "Web Service".
    *   Connect your GitHub account and select your repository.
4.  **Configure**:
    *   **Name**: Give your app a name (e.g., `xray-detection`).
    *   **Runtime**: Select **Python 3**.
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn app:app`
5.  **Deploy**: Click "Create Web Service". Render will build and deploy your app.

## Option 2: Heroku
Heroku is a popular platform for hosting Python apps.

1.  **Install Heroku CLI**: Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).
2.  **Login**: Run `heroku login` in your terminal.
3.  **Create App**: Run `heroku create your-app-name`.
4.  **Deploy**:
    ```bash
    git push heroku main
    ```
5.  **Open**: Run `heroku open` to see your live site.

## Option 3: Docker (Any Cloud Provider)
You can deploy the app as a Docker container to AWS, Google Cloud, or Azure.

1.  **Build the image**:
    ```bash
    docker build -t xray-app .
    ```
2.  **Run locally (to test)**:
    ```bash
    docker run -p 5000:5000 xray-app
    ```
3.  **Push to Container Registry**: Tag and push your image to Docker Hub, AWS ECR, or Google Artifact Registry.
4.  **Deploy**: Use the cloud provider's container service (e.g., AWS App Runner, Google Cloud Run) to deploy the image.

## ⚠️ Important Notes
*   **Model File**: Ensure your `resnet50v2_chest_xray.h5` file is included in your Git repository or Docker image. It is required for the app to work. Note that GitHub has a file size limit of 100MB. If your model is larger, you may need to use **Git LFS** (Large File Storage).
*   **Memory**: Deep learning models can be memory-intensive. If your app crashes on the free tier of cloud providers, you may need to upgrade to a plan with more RAM (e.g., 1GB+).
