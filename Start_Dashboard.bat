@echo off
title Super Lig Prediction Dashboard Launcher
color 0A

echo.
echo  ===============================================
echo  🚀 Super Lig Prediction Dashboard
echo  ===============================================
echo.
echo  📊 Real-time Monte Carlo Simulation Dashboard
echo  🎯 Interactive Turkish Super Lig Predictions
echo.
echo  Features:
echo  • Up to 1,000,000+ simulations
echo  • Real-time probability calculations
echo  • Live statistical analysis
echo  • Interactive charts and confidence intervals
echo.
echo  ===============================================
echo.

python launch_dashboard.py

if errorlevel 1 (
    echo.
    echo ❌ Error occurred. Trying alternative Python command...
    py launch_dashboard.py
)

if errorlevel 1 (
    echo.
    echo ❌ Python not found. Please install Python 3.7+ from python.org
    echo.
    pause
)

echo.
echo Dashboard closed.
pause
