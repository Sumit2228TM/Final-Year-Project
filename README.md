# ⚡ NeuroLink Assist v1.0

> **Adaptive Multi-Modal Assistive System** for individuals with motor, visual, and hearing disabilities.

---

## 🏗️ Architecture

```
neurolink_assist/
├── main.py                  # Entry point — zero-login boot, session orchestration
├── requirements.txt         # Pinned dependencies
├── eye_engine/
│   ├── __init__.py
│   └── eye_engine.py        # MediaPipe Face Mesh gaze + iris + blink + dwell
├── voice_engine/
│   ├── __init__.py
│   └── speech_engine.py     # STT (SpeechRecognition) + TTS (pyttsx3)
├── core_bridge/
│   ├── __init__.py
│   └── cross_comm.py        # Cross-Communication bus (pub/sub message broker)
└── ui/
    ├── __init__.py
    └── app_gui.py            # PyQt6 high-contrast accessible desktop UI
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `eye_engine` | Camera capture → MediaPipe Face Mesh → iris landmark tracking → gaze ratio → calibrated screen coord → exponential smoothing → dwell timer → blink (EAR) detection |
| `voice_engine` | Microphone → SpeechRecognition (Google/Sphinx fallback) → text callback; pyttsx3 queue-driven non-blocking TTS |
| `core_bridge` | Publish/subscribe message bus routing `EYE_SELECTION`, `VOICE_COMMAND`, `TEXT_INPUT` → `DISPLAY_TEXT`, `SPEAK_ALOUD`, `NAVIGATE_UI`, `EMERGENCY` |
| `ui` | PyQt6 dark high-contrast UI with dwell-selectable action grid, camera preview, live captions, mode switcher, 5-point calibration overlay |
| `main.py` | Orchestrates engine lifecycle, bus wiring, calibration flow, mode switching, and clean shutdown |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Webcam** (for eye tracking)
- **Microphone** (for voice commands)
- **Speakers** (for TTS output)
- **Windows 10/11** (primary target; Linux/macOS should work with minor adjustments)

### Installation

```bash
# 1. Clone / navigate to the project
cd neurolink_assist

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

> **Note:** `PyAudio` may require system-level libraries.  
> On Windows, `pip install PyAudio` usually works.  
> On Linux: `sudo apt install portaudio19-dev python3-pyaudio`  
> On macOS: `brew install portaudio && pip install PyAudio`

---

## 🎯 Calibration Guide

The 5-point calibration maps your eye position to screen coordinates:

1. Click the **🎯 Calibrate** button in the header.
2. A full-screen overlay appears with a pulsing amber dot.
3. **Fixate your gaze** on the dot.
4. **Blink deliberately** (hold for ~0.4 seconds) to capture the point.
5. The dot moves to the next position. Repeat for all 5 points:
   - Top-Left → Top-Right → Center → Bottom-Left → Bottom-Right
6. After all 5 points, calibration is complete.

### Fallback Strategies for Poor Lighting

| Condition | Automatic Handling |
|-----------|-------------------|
| Face not detected | Last known gaze position retained with confidence = 0 |
| Low contrast / backlighting | MediaPipe's `min_detection_confidence` is set to 0.5; reduce to 0.3 if needed |
| Partial face occlusion | Iris fallback uses averaged left+right ratios; single-eye still works |
| Camera failure | UI remains functional via Voice and Text modes |
| No calibration | Linear fallback mapping (pupil ratio × screen dimensions) |

---

## 🎮 Operating Modes

### 1. 👁️ Eye Gaze Mode
- Gaze cursor follows your eyes in real-time
- **Dwell Selection**: Hold gaze on a button for 1.5s to activate
- **Blink Click**: Long blink (>400ms) acts as a manual click
- Emergency presets accessible via gaze dwell

### 2. 🎙️ Voice Mode
- Speaks into microphone → text appears in caption area
- **Keyword triggers**: Say "water", "help", "emergency", "pain", "call" etc. to trigger emergency presets
- All recognized text is published to the communication bus

### 3. 📝 Deaf/Caption Mode
- Live captions display all spoken input in high-contrast text
- Visual alerts for emergency events
- STT runs continuously for real-time transcription

### Text Input (All Modes)
- Type in the footer text box and press Enter or click Send
- Message is spoken aloud via TTS and displayed in captions

---

## 🚨 Emergency Presets

| Preset | Trigger (Voice Keywords) | Icon |
|--------|--------------------------|------|
| I need water | "water" | 💧 |
| Emergency SOS | "help", "emergency" | 🚨 |
| Turn off lights | "lights" | 💡 |
| Call caregiver | "call", "caregiver" | 📞 |
| Pain detected | "pain" | ⚠️ |
| Yes | — | ✅ |
| No | — | ❌ |
| Thank you | — | 🙏 |

---

## 🔌 v2.0 Pluggable Interface Blueprint

The modular architecture is designed for seamless extension:

```
v2.0 Additions (plug into existing modules):
─────────────────────────────────────────────
┌──────────────┐     ┌──────────────┐
│  FastAPI      │◄───►│  core_bridge  │   REST/WebSocket endpoints
│  Server       │     │  (bus)        │   wrapping the CrossCommBus
└──────────────┘     └──────────────┘
        │
        ▼
┌──────────────┐
│  AWS S3       │   Session recordings, calibration profiles
│  Storage      │   User preference persistence
└──────────────┘

┌──────────────┐     ┌──────────────┐
│  Firebase     │◄───►│  main.py      │   Remote auth, multi-device sync
│  Auth/DB      │     │  (session)    │
└──────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────┐
│  IoT Relay    │◄───►│  core_bridge  │   Smart home control (lights, locks)
│  (MQTT/HTTP)  │     │  (bus)        │   via emergency presets
└──────────────┘     └──────────────┘
```

### How to Extend

1. **FastAPI endpoint**: Create `api/server.py`, subscribe to `CrossCommBus` events, expose REST + WebSocket.
2. **Cloud storage**: Add `storage/s3_client.py`, hook into session lifecycle in `main.py`.
3. **Authentication**: Add `auth/firebase_auth.py`, wrap `NeuroLinkSession.boot()` with auth check.
4. **IoT**: Add `iot/mqtt_relay.py`, subscribe to `EMERGENCY` messages on the bus.

No core module rewrites required — the bus architecture accepts new subscribers at any time.

---

## 📋 Technical Details

### Gaze Pipeline

```
Camera (30fps) → MediaPipe Face Mesh (468+ landmarks)
    → Iris Center (avg of left [468-472] + right [473-477])
    → Pupil-to-Eye-Corner Ratio (normalized x, y)
    → Affine Calibration Transform (5-point least-squares)
    → Exponential Smoothing (α=0.35)
    → Screen Coordinate (clamped to display bounds)
    → Dwell Timer / Blink Detector → Action Trigger
```

### Threading Model

| Thread | Purpose |
|--------|---------|
| Main (Qt) | UI rendering, event loop, signal dispatch |
| EyeEngine | Camera capture + MediaPipe processing |
| STT | Microphone listening + speech recognition |
| TTS | pyttsx3 synthesis (queue-driven) |
| CommBus | Message dispatch to subscribers |

All engine→UI communication uses Qt's `pyqtSignal` for thread safety.

---

## 📄 License

MIT License — Free for academic and personal use.

---

*Built with ❤️ for accessibility.*
