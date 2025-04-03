# Mining Worker Safety Monitoring System

![Mining Safety Monitor Dashboard](assets/dashboard.png)

## Overview

This project implements an AI-based real-time monitoring system for mining worker safety using data from wearable devices. The system analyzes sensor data to detect potential health and safety risks, including:

- Fatigue
- Harmful gas exposure
- Physical stress/injury risk
- Heat stress
- Multiple concurrent risks

## Features

- Real-time monitoring of multiple workers
- Multi-class risk classification using LSTM deep learning
- Interactive dashboard with live updates
- Historical data tracking and visualization
- Alert system with recommendations for each risk type
- Simulation mode for demonstration purposes

## System Architecture

![System Architecture](assets/system_architecture.png)

The system consists of:
1. Wearable devices collecting sensor data (simulated in this demo)
2. Deep learning model for risk classification
3. Real-time monitoring dashboard

## Installation

1. Clone this repository:
