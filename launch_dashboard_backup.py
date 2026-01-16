"""
Super Lig Prediction Dashboard Launcher
Automated setup and execution script
"""

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'tkinter'  # Usually comes with Python
    ]
    
    print("🔍 Checking required packages...")
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            if package != 'tkinter':  # tkinter comes with Python
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✅ Successfully installed {package}")
                except subprocess.CalledProcessError:
                    print(f"❌ Failed to install {package}")
                    return False
            else:
                print("⚠️  tkinter not available. Please install Python with tkinter support.")
                return False
    
    return True

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def main():
    """Main launcher function"""
    print("🚀 Super Lig Prediction Dashboard Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install required packages")
        input("Press Enter to exit...")
        return
    
    print("\n✅ All requirements satisfied!")
    print("🎯 Launching Interactive Dashboard...")
    print("\nDashboard Features:")
    print("• Real-time Monte Carlo simulation (up to 1M+ simulations)")
    print("• Live probability calculations and statistics")
    print("• Interactive charts and confidence intervals")
    print("• Championship, European, and relegation predictions")
    print("• Detailed statistical analysis")
    
    try:
        # Import and run the dashboard
        from interactive_dashboard import main as run_dashboard
        print("\n🎮 Dashboard starting...")
        run_dashboard()
        
    except ImportError as e:
        print(f"\n❌ Error importing dashboard: {e}")
        print("Make sure 'interactive_dashboard.py' is in the same directory")
        input("Press Enter to exit...")
        
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
