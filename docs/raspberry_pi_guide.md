# Raspberry Pi Deployment Guide

## Overview

This guide covers deploying the cutlery classifier on Raspberry Pi, including setup, optimization, and monitoring.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Initial Setup](#initial-setup)
3. [Model Deployment](#model-deployment)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

## Hardware Requirements

- Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
- SD Card (32GB recommended)
- USB Camera or Pi Camera Module
- Power supply with at least 3A output

## Initial Setup

1. Install dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv
pip3 install -r requirements.txt
```

2. Enable camera:

```bash
sudo raspi-config
# Navigate to Interface Options -> Camera -> Enable
```

3. Verify ONNX Runtime installation:

```bash
python3 -c "import onnxruntime; print(onnxruntime.get_device())"
```

## Model Deployment

1. Export model for Raspberry Pi:

```bash
python scripts/export_model.py --target pi
```

2. Copy model to Raspberry Pi:

```bash
scp models/exports/cutlery_classifier_edge.onnx pi@raspberrypi:/home/pi/cutlery-classifier/models/
```

3. Test inference:

```bash
python scripts/run_inference_on_pi.py --test_dir test_images/
```

## Performance Optimization

### Memory Management

- Enable swap if needed:

```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### CPU Optimization

- Set CPU governor:

```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Model Optimization

- Use lightweight mode for constrained devices:

```bash
python scripts/run_inference_on_pi.py --lightweight
```

## Monitoring

### Resource Usage

Monitor system resources:

```bash
vcgencmd measure_temp  # CPU temperature
free -h               # Memory usage
top                   # Process monitoring
```

### Performance Metrics

Run benchmarks:

```bash
python scripts/run_inference_on_pi.py --benchmark
```

## Troubleshooting

### Common Issues

1. **Out of Memory**

- Enable lightweight mode
- Reduce batch size
- Increase swap space

2. **Slow Inference**

- Check CPU temperature
- Verify CPU governor settings
- Use quantized model

3. **Camera Issues**

- Check permissions
- Verify camera module connection
- Test with `raspistill -o test.jpg`

### Error Messages

1. **ONNX Runtime Error**

```
Solution: Verify ARM build installation:
pip3 install --force-reinstall onnxruntime
```

2. **Memory Error**

```
Solution: Enable lightweight mode and increase swap
```

3. **Camera Error**

```
Solution: Check camera permissions and connection
```

## Advanced Topics

### Custom Optimizations

1. Thread count tuning:

```python
options = ort.SessionOptions()
options.intra_op_num_threads = 4
```

2. Memory preallocation:

```python
options.enable_cpu_mem_arena = False
```

3. Graph optimization:

```python
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### Health Monitoring

Set up basic health monitoring:

```bash
# Add to crontab
*/5 * * * * /home/pi/cutlery-classifier/scripts/monitor_health.sh
```

### Automatic Recovery

Create recovery script:

```bash
#!/bin/bash
if ! pgrep -f "run_inference_on_pi.py" > /dev/null; then
    systemctl restart cutlery_classifier
fi
```
