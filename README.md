![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/github/license/KernFerm/football-vision-analytics-platform)
![Issues](https://img.shields.io/github/issues/KernFerm/football-vision-analytics-platform)
![Pull Requests](https://img.shields.io/github/issues-pr/KernFerm/football-vision-analytics-platform)
![Stars](https://img.shields.io/github/stars/KernFerm/football-vision-analytics-platform?style=social)
![Forks](https://img.shields.io/github/forks/KernFerm/football-vision-analytics-platform?style=social)


# ⚽ Football Vision Analytics Platform

A comprehensive computer vision solution for automated football match analysis with player tracking, tactical insights, and performance metrics.

## ✨ Key Features

| Feature Category | Description |
|-----------------|-------------|
| **Real-time Tracking** | Multi-player detection with YOLO + DeepSORT |
| **Tactical Analysis** | Team formations, passing networks, heatmaps |
| **Performance Metrics** | Distance covered, speed, sprint stats |
| **Video Processing** | Support for 4K match footage at 60fps |
| **Data Export** | CSV, JSON, and video overlays |

## 🚀 Quick Installation

# Football Tracking and Analysis

## Overview

This project is a backend application for tracking and analyzing football (soccer) matches using computer vision and machine learning. It provides a REST API for uploading match videos, tracking players and the ball, assigning teams, analyzing speed and distance, and exporting results. The system leverages YOLO, OpenCV, UMAP, scikit-learn, and other libraries for advanced video analytics.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [How to Install Python & Add to PATH](#how-to-install-python--add-to-path)
- [Project Installation](#project-installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Usage](#api-usage)
- [What the Application Does](#what-the-application-does)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [License](#license)

---

## Features

- Player and ball tracking from video
- Team assignment and analytics
- Speed and distance calculation for players
- Camera movement analysis
- Secure video upload and processing API
- JWT-based authentication
- Modular, extensible codebase

---

## Requirements

- Python 3.8 or higher (tested with Python 3.11.9)
- pip (Python package manager)
- Git (to clone the repository)

---

## How to Install Python & Add to PATH

1. **Download Python:**
	- Go to the [official Python website](https://www.python.org/downloads/).
	- Download the latest version for Windows.
2. **Run the installer:**
	- **Important:** On the first screen, check the box that says **"Add Python to PATH"** before clicking "Install Now".
3. **Verify installation:**
	- Open Command Prompt or PowerShell and run:
	  ```powershell
	  python --version
	  pip --version
	  ```
	- You should see the installed Python and pip versions.

If you forgot to add Python to PATH, rerun the installer and choose "Modify", then select "Add Python to environment variables".

---

## Project Installation

1. **Clone the repository:**
	```powershell
	git clone https://github.com/KernFerm/Football-Vision-Analytics-Platform.git
	cd football_tracking_and_analysis-main
	```
2. **(Recommended) Create and activate a virtual environment:**
	```powershell
	python -m venv .venv
	.venv\Scripts\activate
	```
3. **Install dependencies:**
	```powershell
	pip install -r requirements.txt
	```

---

## Configuration

Before running the application, set the following environment variables. You can create a `.env` file in the project root or set them in your shell:

- `SECRET_KEY` — Secret key for Flask app
- `JWT_SECRET_KEY` — Secret key for JWT authentication
- `UPLOAD_FOLDER` — Path to the folder for uploaded videos (e.g., `uploads`)
- `SQLALCHEMY_DATABASE_URI` — Absolute path to the SQLite database (e.g., `sqlite:///C:/Users/yourname/football_tracking_and_analysis-main/instance/user.db`)

**Example `.env` file:**
```
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
UPLOAD_FOLDER=uploads
SQLALCHEMY_DATABASE_URI=sqlite:///C:/Users/yourname/football_tracking_and_analysis-main/instance/user.db
```

---

## Running the Application

1. **Activate your virtual environment (if using):**
	```powershell
	.venv\Scripts\activate
	```
2. **Run the Flask app:**
	```powershell
	python app.py
	```
3. The API will be available at `http://127.0.0.1:5000/`

---

## API Usage

### Endpoints

- `GET /` — Homepage (health check)
- `POST /api/login` — User login (returns JWT)
- `POST /api/upload-video` — Upload a football match video for analysis (requires JWT)
- `GET /uploads/<filename>` — Download processed/uploaded video

### Example: Uploading a Video

1. **Login to get a JWT:**
	- Send a POST request to `/api/login` with your credentials (see your user setup).
2. **Upload a video:**
	- Send a POST request to `/api/upload-video` with the video file and your JWT in the Authorization header.
3. **Download results:**
	- Access `/uploads/<filename>` to download the processed video or results.

You can use tools like [Postman](https://www.postman.com/) or `curl` for API requests.

---

## What the Application Does

This application provides a backend API for:

- Uploading football match videos
- Detecting and tracking players and the ball using computer vision (YOLO, OpenCV)
- Assigning teams and analyzing player statistics (speed, distance, etc.)
- Exporting processed results for further analysis or visualization

It is designed for integration with a frontend dashboard or for use in analytics workflows. All configuration and secrets are managed via environment variables for security.

---

## Troubleshooting

- **Database errors:** Ensure the `SQLALCHEMY_DATABASE_URI` uses an absolute path and the target directory exists.
- **Missing dependencies:** Run `pip install -r requirements.txt` again in your active environment.
- **Permission errors:** Run your terminal as administrator if you encounter file access issues.
- **Linux install:** Use `install.sh` (Linux only). For Windows, follow the manual steps above.

---


## ⚠️ Model Files Required

**Important:** This project requires certain machine learning model files (such as YOLO weights and keypoint models) to function. These files are **not provided** in this repository due to licensing and file size restrictions.

- You must manually obtain the correct model files (e.g., `yolov8n.pt`, `key_points_pitch_ver2.pt`, etc.) and place them in the `models/` directory at the project root.
- The application will not run correctly without these files.
- Refer to the documentation of the specific models you wish to use for download instructions.

**Example:**

```
football_tracking_and_analysis-main/
	models/
		yolov8n.pt
		key_points_pitch_ver2.pt
```

If you are running tests, note that the `models/` directory may be cleaned up by test scripts. Recreate it and add your model files again before running the application.

---

## Notes

- The `install.sh` script is for Linux systems only.
- For Windows, use the manual installation steps above.
- All secrets and configuration should be set via environment variables or a `.env` file.
- The SQLite database will be created automatically if it does not exist.
- For best results, use high-quality football match videos.

---

## License

This project is licensed under the MIT License.



