# EPICS-Based Bayesian Optimization for Laser-Plasma Acceleration

This repository contains a full-stack, closed-loop automated optimization system designed for ultrafast laser-plasma acceleration experiments. It integrates **hardware control** (via EPICS), **live data acquisition/analysis** (via Thomson Parabola & Image Monitoring), and **machine learning algorithms** (Bayesian Optimization) into a unified GUI-driven application.

⚠️ **Environment Requirements**: This system has been upgraded to a modern PyTorch/BoTorch backend for advanced machine learning capabilities. It requires `botorch`, `torch`, `ophyd`, and `pyepics`.

## Core Components
1. **OptimizerGUI**: The main control panel. Suggests new machine parameters using Bayesian Optimization and controls the hardware via EPICS.
2. **TPAnalyzer**: Automatically processes Thomson Parabola images to calculate the energy spectrum and extract a scalar objective value.
3. **ImageMonitor**: Provides live image feed and tools for defining Regions of Interest (ROI) and Zero Points for the TP Analyzer.
4. **Google Sheets Integration**: Automatically logs experimental parameters and analysis results to a cloud spreadsheet for real-time monitoring and data backup.
5. **Environment-Aware BO Engine**: Powered by `BoTorch`, the system can ingest uncontrollable environmental variables (e.g., laser pointing, shot-to-shot energy) to dramatically reduce noise, offering both "Active Compensation (Known Context)" and "Robust Optimization (Unknown Context)".
6. **Dummy Simulation Generator (`dummy_npz_generator.py`)**: A built-in GUI to safely test the entire closed-loop system offline. It simulates complex physical environments (e.g., prepulse drift) and generates synthetic `.npz` analysis files that the optimizer can read, allowing algorithm validation without hardware access.

## Evolution of the Project
- **v1.1.0 - v1.2.0**: The system was originally separated into `sim/` (Simulation/Dummy PVs) and `exp/` (Experimental Hardware PVs) modes to allow safe development without hardware access.
- **v1.3.0**: Architecture flattened and optimized for production.
- **v2.0.0 (Current)**: Core engine upgraded from `scikit-optimize` to `BoTorch`. Introduced Environment-Aware Bayesian Optimization to handle uncontrollable experimental fluctuations.

## User Manual
For detailed instructions on setting up the ROIs, connecting to EPICS, and running the automated closed-loop, please see the [User Manual (user_manual.md)](user_manual.md).

## Quick Start
```bash
# Ensure you are on Python 3.9
pip install -r requirements.txt
python OptimizerGUI.py
```
