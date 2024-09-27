# AI-Powered Online Exam Guardian System

This project implements an AI-powered exam proctoring system using computer vision techniques to detect and prevent cheating during online exams.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The AI-Powered Online Exam Guardian System is designed to monitor students during online exams, detecting and preventing cheating through advanced computer vision techniques. It uses real-time face and hand landmark detection, as well as eye tracking, to ensure a secure and fair exam environment.

## Features

The system will use your computer's camera to monitor the exam-taker. Two separate threads are used for face analysis and pose estimation:
1. Face System:
- Detects and analyzes facial features, eye movements, and hand positions
- Performs object detection to identify cell phones
- Displays warnings for suspicious activities
2. Pose System:
- Analyzes body posture to detect suspicious behavior
- Uses an ONNX model for inference

## Usage

To use the AI-Powered Online Exam Guardian System, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/kiendoantrung/ai-online-exam-guardian.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the system:
   ```
   python main.py
   ```

4. View the video log in the `data\external\video_log` directory.

