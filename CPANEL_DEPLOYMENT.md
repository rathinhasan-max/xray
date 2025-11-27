# 🚀 cPanel Deployment Guide

Since you have cPanel, deploying is straightforward, but it requires a few specific steps because it's a Python application, not just a static website.

## Step 1: Prepare Your Files
1.  **Delete** the `venv` folder from your computer (if it exists inside the project folder). Do NOT upload the `venv` folder.
2.  **Zip** all the files in your project folder (including `app.py`, `passenger_wsgi.py`, `requirements.txt`, `models/`, `static/`, `templates/`, etc.).
    *   *Make sure `resnet50v2_chest_xray.h5` is inside the `models` folder!*

## Step 2: Upload to cPanel
1.  Log in to your **cPanel**.
2.  Go to **File Manager**.
3.  Create a new folder for your app (e.g., `xray_app`).
4.  **Upload** your zip file into this folder and **Extract** it.

## Step 3: Setup Python App
1.  Go back to the main cPanel dashboard.
2.  Look for **"Setup Python App"** (under Software).
3.  Click **"Create Application"**.
4.  **Python Version**: Select **3.10** (or the latest available version).
5.  **Application Root**: Enter the folder name you created (e.g., `xray_app`).
6.  **Application URL**: Select your domain (e.g., `yourdomain.com/xray`).
7.  **Application Startup File**: Enter `passenger_wsgi.py`.
8.  **Application Entry Point**: Enter `application`.
9.  Click **Create**.

## Step 4: Install Dependencies
1.  After the app is created, you will see a "Configuration File" section in the Python App dashboard.
2.  Enter `requirements.txt` in the box and click **Add**.
3.  Click the **"Run Pip Install"** button.
    *   *This will install Flask, TensorFlow, etc. on the server. It might take a few minutes.*

## Step 5: Restart and Test
1.  Click the **Restart** button for your application.
2.  Visit your URL (e.g., `yourdomain.com/xray`).

## Troubleshooting

### ❌ Error: "Could not open requirements file: No such file or directory"
This means cPanel cannot find your `requirements.txt`. This usually happens if:
1.  You uploaded a folder *inside* a folder (e.g., `xray_app/Gravity Thesis/requirements.txt`).
2.  The file name is wrong.

**✅ Solution: Use the Terminal (Recommended)**
The "Run Pip Install" button can be buggy. It's better to do it manually:

1.  In cPanel, look for **"Terminal"** (under Advanced) and open it.
2.  Type `ls` and press Enter to list your files.
3.  Enter your virtual environment command (copy this from the top of your Python App page, it looks like `source /home/user/virtualenv/.../bin/activate`).
4.  Go to your app folder:
    ```bash
    cd xray_app
    ```
    *(If you see another folder inside, `cd` into that one too until you see `requirements.txt`)*
5.  Run the install command manually:
    ```bash
    pip install -r requirements.txt
    ```
    *If this works, your dependencies are installed!*

### ❌ Internal Server Error
Check the **error log** in cPanel (usually inside the `xray_app` folder or in the main logs section).

### ❌ Model Not Found
Ensure the `models` folder and the `.h5` file were uploaded correctly. The structure should be:
```
xray_app/
├── app.py
├── passenger_wsgi.py
├── requirements.txt
├── models/
│   ├── resnet50v2_chest_xray.h5
│   └── ...
├── static/
└── templates/
```
