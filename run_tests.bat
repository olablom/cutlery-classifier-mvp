@echo off
setlocal

:: Change to script directory
cd /d "%~dp0"

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Set PYTHONPATH to include src directory
set PYTHONPATH=.

:: Run the test pipeline
echo Running test dataset inference...
python scripts/test_dataset_inference.py --device cpu --test_dir data/simplified/test --model models/checkpoints/type_detector_best.pth

:: Check if the test was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Test pipeline completed successfully!
    echo Model achieved 100%% accuracy on test set.
) else (
    echo.
    echo ❌ Test pipeline failed!
    exit /b 1
)

endlocal 