# Troubleshooting Guide

## Overview

This guide helps diagnose and resolve common issues when running the cutlery classifier on Raspberry Pi.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Error Messages](#error-messages)
3. [Performance Problems](#performance-problems)
4. [System Issues](#system-issues)
5. [Recovery Procedures](#recovery-procedures)

## Common Issues

### 1. Model Loading Failures

```
Problem: ONNX Runtime fails to load model
Error: "Failed to load ONNX model: Invalid model"
```

**Solutions:**

1. Verify model export:

```bash
# Re-export model with correct settings
python scripts/export_model.py --target pi --verify
```

2. Check ONNX Runtime installation:

```bash
pip3 install --force-reinstall onnxruntime
```

3. Validate model format:

```python
import onnx
model = onnx.load("models/exports/cutlery_classifier_edge.onnx")
onnx.checker.check_model(model)
```

### 2. Memory Issues

```
Problem: Out of memory during inference
Error: "MemoryError" or system becomes unresponsive
```

**Solutions:**

1. Enable lightweight mode:

```bash
python scripts/run_inference_on_pi.py --lightweight
```

2. Increase swap space:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

3. Monitor memory usage:

```bash
watch -n 1 free -h
```

### 3. Camera Problems

```
Problem: Camera not detected or errors during capture
Error: "Failed to open camera device"
```

**Solutions:**

1. Check camera module:

```bash
vcgencmd get_camera
# Should show: supported=1 detected=1
```

2. Test camera:

```bash
raspistill -o test.jpg
```

3. Verify permissions:

```bash
sudo usermod -a -G video $USER
```

## Error Messages

### 1. ONNX Runtime Errors

```python
# Error: "No valid provider available"
Solution:
import onnxruntime as ort
print(ort.get_available_providers())
# Ensure CPUExecutionProvider is listed
```

### 2. Memory Errors

```python
# Error: "Cannot allocate memory"
Solution:
# Add memory monitoring
def check_memory():
    import psutil
    if psutil.virtual_memory().available < 512 * 1024 * 1024:
        raise MemoryError("Insufficient memory")
```

### 3. Performance Warnings

```python
# Warning: "Inference time exceeds threshold"
Solution:
# Add performance monitoring
def monitor_performance(inference_time):
    if inference_time > 200:  # ms
        logging.warning(f"Slow inference: {inference_time}ms")
```

## Performance Problems

### 1. Slow Inference

**Symptoms:**

- High inference times
- Inconsistent performance
- System lag

**Solutions:**

1. Check CPU temperature:

```bash
vcgencmd measure_temp
```

2. Monitor CPU frequency:

```bash
watch -n 1 vcgencmd measure_clock arm
```

3. Optimize settings:

```python
# In run_inference_on_pi.py
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### 2. High Memory Usage

**Symptoms:**

- System slowdown
- Frequent swapping
- Out of memory errors

**Solutions:**

1. Monitor memory:

```bash
vmstat 1
```

2. Check process memory:

```bash
ps aux | grep python
```

3. Enable memory optimizations:

```python
# In run_inference_on_pi.py
options.enable_cpu_mem_arena = False
```

## System Issues

### 1. Temperature Warnings

**Symptoms:**

- Thermal throttling
- Performance degradation
- System instability

**Solutions:**

1. Monitor temperature:

```bash
#!/bin/bash
while true; do
    temp=$(vcgencmd measure_temp)
    echo "$(date): $temp"
    sleep 5
done
```

2. Improve cooling:

- Add heatsinks
- Ensure proper ventilation
- Consider active cooling

3. Implement thermal management:

```python
def check_temperature():
    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
        temp = float(f.read()) / 1000
        if temp > 80:
            raise RuntimeError(f"Critical temperature: {temp}°C")
```

### 2. Storage Issues

**Symptoms:**

- Slow file operations
- Write errors
- System lag

**Solutions:**

1. Check disk space:

```bash
df -h
```

2. Monitor I/O:

```bash
iostat -x 1
```

3. Clean up logs:

```bash
sudo journalctl --vacuum-time=2d
```

## Recovery Procedures

### 1. Emergency Recovery

If the system becomes unresponsive:

```bash
#!/bin/bash
# recovery.sh
function emergency_recovery() {
    # Kill inference process
    pkill -f run_inference_on_pi.py

    # Clear cache
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

    # Restart service
    sudo systemctl restart cutlery_classifier
}
```

### 2. Automatic Recovery

Set up automatic recovery:

```bash
# In /etc/systemd/system/cutlery_classifier.service
[Unit]
Description=Cutlery Classifier Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/cutlery-classifier
ExecStart=/usr/bin/python3 scripts/run_inference_on_pi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. Health Checks

Implement regular health checks:

```python
def health_check():
    checks = {
        'memory': check_memory(),
        'temperature': check_temperature(),
        'inference': check_inference_time(),
        'camera': check_camera()
    }

    return all(checks.values())
```

## Best Practices

1. **Regular Monitoring**

   - Set up automated health checks
   - Monitor system resources
   - Log performance metrics

2. **Preventive Maintenance**

   - Regular system updates
   - Log rotation
   - Temperature monitoring

3. **Backup Procedures**
   - Regular model backups
   - Configuration backups
   - System state snapshots
