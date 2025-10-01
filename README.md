# Face Service (FastAPI + InsightFace + ONNXRuntime)

A FastAPI service for **face detection and embedding extraction** using [InsightFace](https://github.com/deepinsight/insightface) with ONNXRuntime (CPU or GPU).

## Features
- REST API (`/analyze`) → upload an image, get bounding boxes, detection scores, embeddings.
- Status endpoint (`/status`) → shows backend (CPU/GPU), model bundle, and providers.
- Configurable via environment variables.
- Runs manually with `uvicorn` or as a **systemd service** on Ubuntu.

## Project structure
```text
face-service/
├── app.py                   # FastAPI app
├── requirements-cpu.txt     # Dependencies for CPU build
├── requirements-gpu.txt     # Dependencies for GPU build
├── face.env.example         # Example environment config
├── face-service@.service    # Systemd unit template
├── .venv/                   # Virtual environment (local, ignored in git)
└── README.md                # This file
```

## Installation

### 1. Clone the project
You can clone this repo anywhere (e.g., `/home/<user>/Develop/face-service` or `/opt/face-service`):

```bash
git clone https://github.com/yourusername/face-service.git
cd face-service
```

⚠️ Important: If you move the repo later, update the paths in your .env file (see Configuration).

### 2. Set up environment

Create a Python 3.9+ virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install dependencies (choose one):

CPU build

```bash
pip install -r requirements-cpu.txt
```

GPU build (CUDA)

```bash
pip install -r requirements-gpu.txt
```

### 3. Configuration

Environment is controlled by variables in an .env file.

An example env file is included: face.env.example.

Copy it to /etc/face-service/ and edit as needed:

```bash
sudo mkdir -p /etc/face-service
sudo cp face.env.example /etc/face-service/face.env
sudo nano /etc/face-service/face.env
```

Key values to set:

- APP_DIR → path where you cloned this repo
- VENV_DIR → path to the venv inside this repo (usually <APP_DIR>/.venv)
- PORT → service port (e.g. 8000, 8001 for another instance)
- USE_CPU, GPU_ID, FACE_MODEL → control runtime

## Usage

### 1. Run manually

From inside the project directory:

```bash
source .venv/bin/activate
python app.py
```

Or run with uvicorn (recommended):

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Run as a systemd service (Ubuntu)

This repo includes a systemd unit template: face-service@.service.

#### 2.1 Copy to systemd
```bash
sudo cp face-service@.service /etc/systemd/system/
```

#### 2.2 Reload and enable
```bash
sudo systemctl daemon-reload
sudo systemctl enable face-service@lindo
sudo systemctl start face-service@lindo
```

Replace lindo with your Linux username.

The service will read /etc/face-service/face.env for settings.

#### 2.3 Check logs
```bash
systemctl status face-service@lindo
journalctl -u face-service@lindo -f
```

### 3. API usage

Check service status

```bash
curl http://127.0.0.1:8000/status
```



Analyze an image

```bash
curl -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/analyze
```
