<p align="center">
  <img src="frontend/src/assets/cvacar.svg" alt="ClassVision-ACAR Logo" width="400"/>
</p>

<h1 align="center">ClassVision — Advanced Classroom Activity Recognition</h1>

<p align="center">
  <b>AI-powered real-time student activity monitoring for smarter classrooms</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Django-5.1-092E20?style=for-the-badge&logo=django&logoColor=white" alt="Django"/>
  <img src="https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10-00897B?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-FF6F61?style=for-the-badge" alt="YOLO"/>
</p>

> **Academic Project** — B.Tech (Computer Science & Engineering — Data Science), VI Semester  
> **Institution** — Shri Ramdeobaba College of Engineering & Management, Nagpur  
> **Guide** — Dr. Uma Yadav, Department of CSE & Emerging Technologies

---

## 📋 Table of Contents

- [What is ClassVision?](#-what-is-classvision)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [System Architecture](#-system-architecture)
- [Activities Detected](#-activities-detected)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#1-backend-setup)
  - [Frontend Setup](#2-frontend-setup)
- [API Reference](#-api-reference)
- [ML Pipeline Deep Dive](#-ml-pipeline-deep-dive)
- [Model Performance & Results](#-model-performance--results)
- [Dashboard & Analytics](#-dashboard--analytics)
- [Screenshots](#-screenshots)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [Team](#-team)
- [References](#-references)
- [License](#-license)

---

## 🎯 What is ClassVision?

**ClassVision-ACAR** (Advanced Classroom Activity Recognition) is a full-stack AI application that empowers educators with **real-time insights** into what students are doing in the classroom.

Upload a classroom video or connect a live camera feed — ClassVision will automatically:

1. **Detect** every student using YOLOv11 object detection
2. **Track** each individual across frames with persistent ID assignment
3. **Classify** their activity using pose-estimation + deep learning
4. **Identify** students by face recognition for personalized logging
5. **Visualize** everything on an interactive analytics dashboard

> **In Simple Terms:** Think of it as a smart classroom assistant that watches a video of your class, figures out what each student is doing (reading, sleeping, using their phone, etc.), and gives you a clean report with charts and stats.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎬 **Video Upload & Processing** | Upload any classroom recording — get back an annotated video with activity labels on each student |
| 📡 **Live Monitoring** | Connect an IP camera or webcam for real-time activity stream |
| 🧠 **7 Activity Classes** | Detects eating, hand raising, reading, sitting, sleeping, writing, and phone usage |
| 👤 **Face Recognition** | Identifies registered students by face and logs their activities by name |
| 📊 **Analytics Dashboard** | Interactive charts — bar graphs, pie charts, line charts, and KPI cards |
| 🔐 **Authentication System** | Full user registration, login, OTP email verification, and token-based auth |
| 📱 **Phone Detection** | Special YOLO-based phone detection — flags the closest person as "Using Phone" |
| 📝 **CSV Activity Logs** | Every detected activity is timestamped and logged to a CSV for offline analysis |

---

## 🔄 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS VIDEO                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. YOLO v11 — Detect all persons + phones in each frame           │
│     └── Assigns persistent tracking IDs to each student            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────────┐
          │  Phone Detected │   │  No Phone Detected  │
          │  near person?   │   │  (Normal pipeline)  │
          └────────┬────────┘   └──────────┬──────────┘
                   │                       │
                   ▼                       ▼
          Label: "Using Phone"   ┌─────────────────────────────┐
                                 │ 2. Crop each person ROI     │
                                 │    Resize to 224×224        │
                                 └─────────────┬───────────────┘
                                               │
                                               ▼
                                 ┌─────────────────────────────┐
                                 │ 3. MediaPipe Holistic       │
                                 │    Extract 1662 keypoints   │
                                 │    (pose + face + hands)    │
                                 └─────────────┬───────────────┘
                                               │
                                               ▼
                                 ┌─────────────────────────────┐
                                 │ 4. Buffer 30 frames of      │
                                 │    keypoints per student    │
                                 └─────────────┬───────────────┘
                                               │
                                               ▼
                                 ┌─────────────────────────────┐
                                 │ 5. LSTM Model predicts      │
                                 │    activity class           │
                                 └─────────────┬───────────────┘
                                               │
                               ┌───────────────┴────────────────┐
                               ▼                                ▼
                    ┌───────────────────┐            ┌───────────────────┐
                    │ 6. Face Recog.    │            │ 7. Draw bounding  │
                    │    Identify who   │            │    boxes + labels │
                    │    is doing what  │            │    on video frame │
                    └───────────────────┘            └───────────────────┘
                               │                                │
                               ▼                                ▼
                    ┌───────────────────┐            ┌───────────────────┐
                    │ 8. Log to CSV     │            │ 9. Output video   │
                    │   (Name, Action,  │            │   returned to     │
                    │    Timestamp)     │            │   frontend        │
                    └───────────────────┘            └───────────────────┘
```

---

## 🏗 System Architecture

```
ClassVision-ACAR/
├── frontend/          →  React + Vite SPA (user interface)
├── backend/           →  Django REST API (business logic + ML inference)
└── ml/                →  ML experiments, training scripts, model weights
```

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Vite)                  │
│                                                              │
│   Landing Page  ←→  Login/Signup  ←→  Upload  ←→  Dashboard │
│                                                              │
│   • TailwindCSS + Parallax effects                          │
│   • Recharts for analytics visualization                    │
│   • Axios for API communication                             │
│   • React Router for SPA navigation                         │
└───────────────────────┬──────────────────────────────────────┘
                        │  HTTP / REST
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  BACKEND (Django + DRF)                       │
│                                                              │
│   /auth/*          →  Registration, Login, OTP, Tokens       │
│   /classify/*      →  Video upload, Live stream processing   │
│   /analysis/*      →  Dashboard KPIs, charts, statistics     │
│                                                              │
│   Models: YOLO v11 + MediaPipe + LSTM (TensorFlow/Keras)     │
│   DB: SQLite  |  Auth: Token-based  |  Email: SMTP/Gmail    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🏷 Activities Detected

The system classifies student behavior into **7 categories**, split into productive and distracted:

### ✅ Productive Activities
| Activity | What It Means |
|---|---|
| ✋ **Hand Raise** | Student is raising their hand (participating) |
| 📖 **Reading Book** | Student is reading from a textbook |
| 🪑 **Sitting on Desk** | Student is sitting attentively (default posture) |
| ✍️ **Writing in Textbook** | Student is writing or taking notes |

### ❌ Distracted Activities
| Activity | What It Means |
|---|---|
| 🍕 **Eating in Classroom** | Student is eating during class |
| 😴 **Sleeping** | Student is sleeping or has head down |
| 📱 **Using Phone** | Phone detected near a student (YOLO-based) |

### 👨‍🏫 Teacher Activities (also modelled)
| Activity | What It Means |
|---|---|
| 🗣️ **Explaining the Subject** | Teacher is actively delivering lessons or interacting with students |
| 📱 **Using Mobile Phone** | Teacher is using their phone during class |
| ✍️ **Writing on Board** | Teacher is writing or drawing on the blackboard/whiteboard |

> **Color Coding in Video Output:**
> - 🟩 **Green** bounding box → Productive activity
> - 🟥 **Red** bounding box → Distracted activity

---

## 🛠 Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| **React 18** | Component-based UI framework |
| **Vite 6** | Fast build tool and dev server |
| **TailwindCSS 3** | Utility-first CSS styling |
| **React Router 7** | Client-side routing |
| **Recharts** | Dashboard charts (Bar, Line, Pie, Donut) |
| **Axios** | HTTP client for API calls |
| **react-just-parallax** | Parallax scrolling effects on landing page |
| **react-toastify** | Toast notifications |

### Backend
| Technology | Purpose |
|---|---|
| **Django 5.1** | Web framework |
| **Django REST Framework** | API layer |
| **SQLite** | Database (development) |
| **Token Authentication** | Session management via DRF auth tokens |

### Machine Learning
| Technology | Purpose |
|---|---|
| **YOLOv11 (Ultralytics)** | Real-time person + phone detection & tracking |
| **MediaPipe Holistic** | Full-body keypoint extraction (pose, face, hands) |
| **TensorFlow / Keras** | LSTM model for temporal activity classification |
| **face_recognition (dlib)** | Student identification via facial encoding |
| **OpenCV** | Video I/O, frame processing, annotation |
| **NumPy / Pandas** | Data manipulation and CSV logging |
| **FFmpeg** | Video transcoding to streamable MP4 |

---

## 📂 Dataset

### Edunet DRSTA Dataset

The primary dataset used is the **Edunet DRSTA** (Digital Repository for Smart Teaching and Assessment) Dataset, provided by the Edunet Foundation for academic research use.

| Specification | Value |
|---|---|
| Total videos | 7,851 |
| Student action clips | 4,228 |
| Teacher action clips | 3,623 |
| Total classroom action classes | 9 |
| Student action classes | 6 |
| Teacher action classes | 3 |
| Clip duration | 3.25 – 12.7 seconds |
| Total footage duration | ~12 hours |
| Students per clip | 2 – 6 |

The dataset was captured in authentic classroom settings using overhead or corner-mounted surveillance cameras. Each video is annotated at the **clip level** (not per-student), which presents a challenge for per-student classification models.

### Custom Dataset

To supplement the Edunet dataset and address gaps in coverage, a **custom dataset** was created by the team. It adds two activity classes absent from Edunet:

- **Sleeping in classroom** — head-down posture, indicating disengagement
- **Using mobile phone** — small repetitive hand/wrist movements near the torso

The custom recordings were produced in controlled settings while maintaining format consistency with the Edunet dataset, increasing overall data diversity and action granularity.

---

## 📁 Project Structure

```
ClassVision-ACAR/
│
├── backend/                          # Django Backend
│   ├── manage.py                     # Django management script
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── classvision/                  # Django project settings
│   │   ├── settings.py               # App config, DB, CORS, email
│   │   ├── urls.py                   # Root URL routing
│   │   ├── wsgi.py                   # WSGI entry point
│   │   └── asgi.py                   # ASGI entry point
│   │
│   ├── authenticate/                 # User authentication app
│   │   ├── models.py                 # Custom User model (roles, OTP, phone)
│   │   ├── managers.py               # Custom user manager
│   │   ├── views.py                  # Register, Login, Logout, OTP, Token APIs
│   │   ├── decorators.py             # Role-based access control decorator
│   │   └── urls.py                   # Auth URL routes
│   │
│   ├── classification/               # Video processing + ML inference
│   │   ├── views.py                  # Upload & live video endpoints, ML pipeline
│   │   ├── urls.py                   # Classification URL routes
│   │   ├── weights/                  # Pre-trained model weights (.h5, .pt)
│   │   └── encodings.pickle          # Pre-computed face encodings
│   │
│   └── analysis/                     # Analytics & dashboard data
│       ├── models.py                 # ClassData model (CSV storage)
│       ├── views.py                  # KPI, activity count, stats APIs
│       └── urls.py                   # Analysis URL routes
│
├── frontend/                         # React Frontend
│   ├── package.json                  # Node.js dependencies
│   ├── vite.config.js                # Vite configuration
│   ├── tailwind.config.js            # Tailwind theme configuration
│   ├── index.html                    # HTML entry point
│   │
│   └── src/
│       ├── App.jsx                   # Root component with routing
│       ├── main.jsx                  # React DOM entry point
│       ├── index.css                 # Global styles
│       ├── dashboard.css             # Dashboard-specific styles
│       │
│       ├── components/
│       │   ├── Hero.jsx              # Landing page hero section
│       │   ├── Header.jsx            # Navigation bar
│       │   ├── Footer.jsx            # Page footer
│       │   ├── Login.jsx             # Login form
│       │   ├── Signup.jsx            # Registration form
│       │   ├── Upload.jsx            # Video upload + processing UI
│       │   ├── Live.jsx              # Live camera monitoring UI
│       │   ├── Dashboard.jsx         # Analytics dashboard with charts
│       │   ├── Benefits.jsx          # Feature cards section
│       │   ├── Collaboration.jsx     # "How to use" section
│       │   ├── Button.jsx            # Reusable button component
│       │   ├── Section.jsx           # Reusable page section wrapper
│       │   ├── Heading.jsx           # Section heading component
│       │   ├── Notification.jsx      # Notification bubble component
│       │   ├── Generating.jsx        # Loading indicator
│       │   ├── CompanyLogos.jsx      # Partner/logo strip
│       │   ├── Tagline.jsx           # Tagline text component
│       │   └── design/              # Design utility components (gradients, curves)
│       │
│       ├── constants/
│       │   └── index.js              # Navigation items, feature cards, social links
│       │
│       └── assets/                   # Images, icons, SVGs
│
└── ml/                               # ML Experiments & Training
    ├── requirements.txt              # ML-specific dependencies
    ├── students_final_model.py       # Final multi-student inference pipeline
    ├── data_pipeline_lrcn.py         # LRCN-based activity recognition
    ├── data_pipeline_pose_estimation.py  # Pose-based activity recognition
    ├── pose_estimation_inference.py   # Single-person pose inference
    ├── face_encode.py                # Face encoding generator
    ├── tracker.py                    # FFmpeg video converter utility
    ├── images/                       # Student face images for recognition
    └── weights/                      # Trained model weights
        ├── actionsIncludingSleeping.h5   # Primary LSTM model (7 classes)
        ├── HandRaise.h5                  # Hand raise detector
        ├── Reading_Book.h5               # Reading detector
        ├── Sitting_on_Desk.h5            # Sitting detector
        ├── Writting_on_Textbook.h5       # Writing detector
        ├── Eating_in_classroom.h5        # Eating detector
        ├── teachers.h5                   # Teacher activity model
        ├── LRCN_demo.h5                  # LRCN demo model
        └── yolo11n.pt                    # YOLOv11 nano weights
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **FFmpeg** (for video transcoding)
- **Git**

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/Pranay-Rokade/ClassVision-ACAR.git
cd ClassVision-ACAR/backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requriments.txt

# Create a .env file with your credentials
cat <<EOF > .env
DJANGO_SECRET_KEY=your-secret-key-here
EMAIL_ID=your-email@gmail.com
EMAIL_PASS=your-app-password
EOF

# Run database migrations
python manage.py makemigrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

> The backend will be running at `http://127.0.0.1:8000`

### 2. Frontend Setup

```bash
# Open a new terminal
cd ClassVision-ACAR/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

> The frontend will be running at `http://localhost:5173`

### 3. Add Student Faces (Optional)

To enable face recognition, place student photos in `ml/images/` with the filename as the student's name:

```
ml/images/
├── Pranay.png
├── Utkarsh.png
└── Yash.png
```

Then generate face encodings:

```bash
cd ml
python face_encode.py
```

### 4. Deployment Notes

The application can be containerized using **Docker** for consistent performance across different systems. The core detection and classification modules support **GPU acceleration** for real-time inference, with **CPU fallbacks** implemented for broader accessibility on standard school hardware.

---

## 📡 API Reference

### Authentication (`/auth/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Register a new user |
| `POST` | `/auth/login` | Login with email & password |
| `POST` | `/auth/logout` | Logout and destroy token |
| `POST` | `/auth/verify-email` | Verify email with OTP |
| `POST` | `/auth/verify-token` | Check if auth token is still valid |
| `POST` | `/auth/<case>/resend-otp` | Resend OTP (email/login) |

### Classification (`/classify/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify/videoclassification` | Upload video → get processed video back |
| `POST` | `/classify/livevideo` | Stream live video with activity annotations |

### Analytics (`/analysis/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/analysis/kpis` | Total students, activities, positive/negative counts |
| `GET` | `/analysis/activity-count` | Count of each activity type |
| `GET` | `/analysis/positive-negative-stats` | Aggregate positive vs negative activities |
| `GET` | `/analysis/activities-per-student` | Activity count per identified student |
| `GET` | `/analysis/percentage-of-actions` | Percentage breakdown of each action |

---

## 🧪 ML Pipeline Deep Dive

### Model Architecture

The primary classification model is an **LSTM (Long Short-Term Memory)** network that operates on **temporal sequences of body keypoints**:

```
Input: 30 frames × 1662 keypoints per frame
       ↓
  ┌────────────────────┐
  │   LSTM Layers      │  ← Learns temporal patterns in body movement
  └────────┬───────────┘
           ↓
  ┌────────────────────┐
  │   Dense + Softmax  │  ← Outputs probability for each of 7 classes
  └────────┬───────────┘
           ↓
  Output: Activity prediction
```

### Keypoint Extraction (1662 features per frame)

| Body Region | Landmarks | Features |
|---|---|---|
| **Pose** | 33 landmarks × 4 values (x, y, z, visibility) | 132 |
| **Face** | 468 landmarks × 3 values (x, y, z) | 1404 |
| **Left Hand** | 21 landmarks × 3 values (x, y, z) | 63 |
| **Right Hand** | 21 landmarks × 3 values (x, y, z) | 63 |
| **Total** | | **1662** |

### Phone Detection (Special Pipeline)

Phone detection doesn't use pose estimation — instead:

1. **YOLO detects** objects of class `67` (cell phone) in the frame
2. **Manhattan distance** is calculated between each phone and each person
3. The **closest person** to each phone is labeled as "Using Phone"
4. That person is **excluded** from the normal pose-estimation pipeline

### Earlier Approach: LRCN Model

Before settling on the Pose + LSTM pipeline, **LRCN (Long-term Recurrent Convolutional Networks)** was explored. LRCN combines CNNs for spatial feature extraction with LSTM layers for temporal modelling, processing sequences of 20 raw image frames (64×64 ROI crops) per student.

While LRCN showed initial promise, it was replaced due to:
- Overfitting — validation loss diverged from training loss after epoch 10
- Poor generalization to unseen classroom conditions (lighting, clutter, occlusion)
- Higher computational cost, making real-time multi-student inference impractical

### Inference Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | Number of frames buffered before prediction |
| `THRESHOLD` | 0.4 | Minimum confidence to accept a prediction |
| `FRAME_INTERVAL` | 6 | Process every 6th frame for efficiency |
| `RESIZE` | 224 × 224 | Crop size for person ROI before keypoint extraction |
| `BBOX_SMOOTHING` | Rolling average | Smooths bounding box jitter across frames |
| `BBOX_EXPANSION` | 1.2× | Expands crop to capture full body |

---

## 📈 Model Performance & Results

### Comparative Analysis

Two model architectures were evaluated for student activity classification:

| Architecture | Training Accuracy | Training Loss |
|---|---|---|
| **LRCN** (Long-term Recurrent Convolutional Network) | 74.4% | ~43.4% |
| **Pose Estimation + Stacked LSTM** ✅ *(selected)* | **97.04%** | **~28%** |

### Why Pose + LSTM Won

- **LRCN** struggled with crowded scenes — CNN spatial features were diluted by background noise and multiple overlapping students. Validation loss increased after epoch 10, a clear sign of overfitting.
- **Pose + LSTM** operates on clean, noise-free skeletal keypoints isolated per student via bounding box tracking. This makes it robust to lighting variation, different clothing, and partial occlusion. Training and validation accuracy both followed a stable upward trajectory across epochs, with validation loss stabilizing — indicating good generalization to unseen data.

---

## 📊 Dashboard & Analytics

The dashboard provides **4 KPI cards** and **4 chart types**:

### KPI Cards
| Card | Metric |
|---|---|
| 🔵 **Students** | Total unique students detected |
| 🟠 **Activities Performed** | Number of distinct activity types |
| 🟢 **Positive Activities** | Count of productive behaviors |
| 🔴 **Negative Activities** | Count of distracted behaviors |

### Charts
| Chart | Visualization |
|---|---|
| 📊 **Bar Chart** | Number of times each activity was detected |
| 📈 **Line Chart** | Activity count per identified student |
| 🥧 **Pie Chart** | Percentage distribution of all activities |
| 🍩 **Donut Chart** | Productive vs Distracted ratio |

---

## 🖼 Screenshots

### User Flow

```
Landing Page  →  Login  →  Upload Video  →  Processing  →  View Result  →  Dashboard
```

| Page | Description |
|---|---|
| **Landing Page** | Hero section with parallax, feature cards, "How to use" guide |
| **Login / Signup** | Gradient-bordered forms with email & password |
| **Video Upload** | Drag & drop video file, process, and view annotated result |
| **Live Monitor** | Real-time camera feed with activity log sidebar |
| **Dashboard** | 4 KPI cards + 4 interactive charts |

---

## 🔭 Future Work

The following enhancements are planned for future iterations of ClassVision:

1. **Emotion & Facial Expression Recognition** — Combine facial expression analysis (confusion, boredom, interest) with the existing pose-based pipeline for a richer behavioral model and more accurate engagement analysis.

2. **Real-Time Alerts** — Trigger automatic notifications for educators when specific patterns are detected — e.g., prolonged phone use, widespread sleeping, or a sudden drop in participation — enabling immediate intervention.

3. **Scalability to Larger Classrooms** — Optimize tracking algorithms and introduce distributed processing to support larger student populations and varied camera layouts without performance degradation.

4. **Expanded & More Diverse Dataset** — Collect training data from multiple schools, age groups, and cultural settings to reduce model bias and improve generalization across real-world conditions.

5. **Multi-Modal Learning & Cognitive Load Analysis** — Integrate gaze tracking, posture analysis, and interaction data to assess student cognitive load — identifying whether students are overwhelmed or under-stimulated to assist in lesson planning.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "Add my feature"`
4. **Push** to the branch: `git push origin feature/my-feature`
5. **Open** a Pull Request

---

## 👥 Team

| Member | Roll No. | Role |
|---|---|---|
| **Ashwin Gour** | 31 | Developer |
| **Pranay Rokade** | 49 | Project Lead |
| **Utkarsh Karambhe** | 64 | Developer |
| **Vivek Sharma** | 67 | Developer |
| **Yash Tapre** | 72 | Developer |

**Guide:** Dr. Uma Yadav — Department of Computer Science & Engineering and Emerging Technologies, RCOEM Nagpur

---

## 📚 References

1. R. Yuvaraj, A. A. Prince, and M. Murugappan, "An automated recognition of teacher and student activities in the classroom environment: A deep learning framework," *IEEE Access*, DOI: 10.1109/ACCESS.2024.3518577. [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10804154)

2. V. Sharma, M. Gupta, A. Kumar, and D. Mishra, "STAR-3D: A Holistic Approach for Human Activity Recognition in the Classroom Environment," *Information*, vol. 15, no. 4, p. 179, 2024. [Link](https://www.researchgate.net/publication/379283201_STAR3D_A_Holistic_Approach_for_Human_Activity_Recognition_in_the_Classroom_Environment)

3. R. Raj and A. Kos, "An improved human activity recognition technique based on convolutional neural network," *Scientific Reports*, vol. 13, p. 22581, 2023.

4. MediaPipe. "MediaPipe Pose." Google Developers. [Link](https://developers.google.com/mediapipe/solutions/vision/pose)

5. J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proc. IEEE CVPR*, 2016, pp. 779–788, DOI: 10.1109/CVPR.2016.91. [Link](https://ieeexplore.ieee.org/document/7780460)

---

## 📄 License

This project is developed for academic/research purposes.

---

<p align="center">
  <b>Built with ❤️ using AI, Computer Vision, and Deep Learning</b>
</p>