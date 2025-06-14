@echo off
setlocal enabledelayedexpansion

:: Check if the script is running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c %~fnx0' -Verb RunAs"
    exit /b
)

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Downloading and installing Python 3.11.9...

    :: Define Python installer URL and file name
    set PYTHON_INSTALLER=python-3.11.9-amd64.exe
    set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/%PYTHON_INSTALLER%

    :: Download Python installer
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_INSTALLER%')"

    :: Install Python silently with admin privileges
    start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1

    :: Remove installer after installation
    del %PYTHON_INSTALLER%

    :: Refresh environment variables
    set PATH=%PATH%;C:\Program Files\Python311\Scripts\;C:\Program Files\Python311\
)

:: Ensure Python is accessible
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Failed to install Python. Exiting...
    exit /b 1
)

:: Change directory to the location where this script (run.bat) is located
cd /d "%~dp0"

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies globally
if exist requirements.txt (
    echo Installing dependencies globally...
    python -m pip install --no-warn-script-location --user -r requirements.txt
) else (
    echo WARNING: requirements.txt not found! Installing Flask manually...
    python -m pip install --user flask
)

:: Run the application
echo Running app.py...
python app.py
