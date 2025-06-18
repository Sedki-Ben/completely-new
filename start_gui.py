#!/usr/bin/env python3
"""
WiFi Indoor Localization GUI Startup Script
==========================================

This script starts the Flask API server and opens the GUI in a web browser.
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask', 'tensorflow', 'numpy', 'pandas', 
        'scikit-learn', 'joblib', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def start_flask_server():
    """Start the Flask API server."""
    print("🚀 Starting Flask API server...")
    
    # Check if localization_api.py exists
    api_file = Path("localization_api.py")
    if not api_file.exists():
        print("❌ localization_api.py not found!")
        print("Please ensure the API file is in the current directory.")
        return False
    
    try:
        # Start Flask server
        subprocess.run([
            sys.executable, "localization_api.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Flask server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True

def open_browser():
    """Open the GUI in a web browser."""
    time.sleep(3)  # Wait for server to start
    print("🌐 Opening GUI in web browser...")
    
    try:
        webbrowser.open('http://localhost:5000')
        print("✅ GUI opened successfully!")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print("Please manually open: http://localhost:5000")

def main():
    """Main function to start the GUI system."""
    print("=" * 60)
    print("🌐 WiFi Indoor Localization System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\n📋 System Information:")
    print(f"   - Python version: {sys.version}")
    print(f"   - Working directory: {os.getcwd()}")
    print(f"   - API file: {'✅ Found' if Path('localization_api.py').exists() else '❌ Missing'}")
    print(f"   - GUI file: {'✅ Found' if Path('indoor_localization_gui.html').exists() else '❌ Missing'}")
    
    print("\n🚀 Starting system...")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask server
    start_flask_server()

if __name__ == "__main__":
    main() 