@echo off
REM Cutlery Classifier MVP - Windows Development Commands
REM =======================================================

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="install-deps" goto install-deps
if "%1"=="test" goto test
if "%1"=="test-quick" goto test-quick
if "%1"=="benchmark" goto benchmark
if "%1"=="benchmark-cpu" goto benchmark-cpu
if "%1"=="stress-test" goto stress-test
if "%1"=="inference" goto inference
if "%1"=="inference-all" goto inference-all
if "%1"=="gradcam" goto gradcam
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="clean" goto clean
if "%1"=="validate" goto validate
if "%1"=="version" goto version

echo Unknown command: %1
goto help

:help
echo Cutlery Classifier MVP - Development Commands
echo =============================================
echo.
echo Setup ^& Installation:
echo   install          Install package in development mode
echo   install-deps     Install dependencies only
echo.
echo Development ^& Testing:
echo   test             Run full test suite
echo   test-quick       Run quick tests only
echo   lint             Run code linting
echo   format           Format code with black
echo.
echo Performance ^& Benchmarking:
echo   benchmark        Run RTX 5090 performance benchmark
echo   benchmark-cpu    Run CPU benchmark for comparison
echo   stress-test      Run stress testing on model
echo.
echo Inference ^& Demo:
echo   inference        Run demo inference on sample image
echo   inference-all    Test inference on all demo classes
echo   gradcam          Generate Grad-CAM visualization
echo.
echo Maintenance:
echo   clean            Clean temporary files and cache
echo   validate         Quick validation that everything works
echo   version          Show version information
echo.
echo Usage: make.bat [command]
goto end

:install
echo Installing Cutlery Classifier in development mode...
pip install -e .
goto end

:install-deps
echo Installing dependencies...
pip install -r requirements.txt
goto end

:test
echo Running full test suite...
python -m pytest tests/ -v --cov=src/cutlery_classifier
goto end

:test-quick
echo Running quick tests...
python -m pytest tests/ -v -k "not slow"
goto end

:benchmark
echo Running RTX 5090 Performance Benchmark...
echo =========================================
python benchmark_rtx5090.py
goto end

:benchmark-cpu
echo Running CPU Benchmark for comparison...
python scripts/run_inference.py --device cpu --image data/raw/fork/IMG_0941[1].jpg
goto end

:stress-test
echo Running stress testing...
python scripts/test_dataset_inference.py --device cuda --test_dir data/simplified/test --stress-test
goto end

:inference
echo Running demo inference...
python scripts/run_inference.py --device cuda --image data/raw/fork/IMG_0941[1].jpg
goto end

:inference-all
echo Testing inference on all classes...
echo Fork:
python scripts/run_inference.py --device cuda --image data/raw/fork/IMG_0941[1].jpg --quiet
echo Knife:
python scripts/run_inference.py --device cuda --image data/raw/knife/IMG_0959[1].jpg --quiet
echo Spoon:
python scripts/run_inference.py --device cuda --image data/raw/spoon/IMG_0962[1].jpg --quiet
goto end

:gradcam
echo Generating Grad-CAM visualization...
python scripts/run_inference.py --device cuda --image data/raw/fork/IMG_0941[1].jpg --grad-cam
goto end

:lint
echo Running code linting...
flake8 src/ scripts/ --max-line-length=88 --ignore=E203,W503
goto end

:format
echo Formatting code...
black src/ scripts/ --line-length=88
goto end

:clean
echo Cleaning temporary files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del "%%f"
if exist .coverage del .coverage
if exist .pytest_cache rd /s /q .pytest_cache
goto end

:validate
echo Quick validation of installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from src.cutlery_classifier.models.model_factory import create_model; print('Model factory works')" 2>nul || echo Model factory needs PYTHONPATH or install -e .
echo Validation complete!
goto end

:version
echo Current version info:
git describe --tags --abbrev=0 2>nul || echo No tags found
for /f "delims=" %%i in ('git log -1 --format^="%%h %%s"') do echo Latest commit: %%i
goto end

:end 