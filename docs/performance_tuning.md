# Performance Tuning Guide

## Overview

This guide covers performance optimization techniques for the cutlery classifier on Raspberry Pi, focusing on memory usage, inference speed, and system stability.

## Table of Contents

1. [Memory Optimization](#memory-optimization)
2. [Model Quantization](#model-quantization)
3. [System Tuning](#system-tuning)
4. [Monitoring & Profiling](#monitoring--profiling)
5. [Benchmarking](#benchmarking)

## Memory Optimization

### 1. Model Memory Reduction

```python
# Enable memory-efficient inference
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.intra_op_num_threads = 4
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

### 2. Input Pipeline Optimization

```python
# Efficient image loading
def load_image(path):
    # Use grayscale to reduce memory
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Preallocate numpy array
    resized = np.empty((320, 320), dtype=np.uint8)
    cv2.resize(image, (320, 320), dst=resized)
    return resized

# Memory-efficient preprocessing
def preprocess_batch(images, batch_size=4):
    # Preallocate batch array
    batch = np.empty((batch_size, 1, 320, 320), dtype=np.float32)
    for i, img in enumerate(images):
        batch[i, 0] = img / 255.0
    return batch
```

### 3. Memory Monitoring

```python
def check_memory():
    import psutil
    mem = psutil.virtual_memory()
    if mem.available < 512 * 1024 * 1024:  # 512MB
        return "low"
    return "ok"
```

## Model Quantization

### 1. Post-Training Quantization

```python
def quantize_model(model_path):
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic

    # Quantize to INT8
    quantized_path = model_path.replace('.onnx', '_quantized.onnx')
    quantize_dynamic(
        model_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    return quantized_path
```

### 2. Quantization-Aware Training

```python
def prepare_qat(model):
    # Enable quantization-aware training
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)
```

### 3. Mixed Precision

```python
def enable_mixed_precision():
    # Use FP16 where beneficial
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options
```

## System Tuning

### 1. CPU Governor Settings

```bash
#!/bin/bash
# Set performance governor
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Monitor temperature
watch -n 1 vcgencmd measure_temp
```

### 2. Memory Configuration

```bash
#!/bin/bash
# Optimize memory split
sudo raspi-config nonint do_memory_split 16

# Configure swap
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Process Priority

```bash
#!/bin/bash
# Set high priority for inference process
sudo nice -n -20 python scripts/run_inference_on_pi.py
```

## Monitoring & Profiling

### 1. Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.temperature = []

    def record_inference(self, time_ms):
        self.inference_times.append(time_ms)

    def record_system_stats(self):
        import psutil
        self.memory_usage.append(psutil.virtual_memory().percent)

        # Get CPU temperature
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000
            self.temperature.append(temp)
```

### 2. Resource Logging

```python
def setup_logging():
    import logging
    logging.basicConfig(
        filename='performance.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
```

## Benchmarking

### 1. Performance Benchmarking

```python
def run_benchmark(model_path, test_images, iterations=50):
    results = {
        'inference_times': [],
        'memory_usage': [],
        'temperature': []
    }

    monitor = PerformanceMonitor()

    for _ in range(iterations):
        for image in test_images:
            # Run inference
            start_time = time.perf_counter()
            prediction = run_inference(image)
            inference_time = (time.perf_counter() - start_time) * 1000

            # Record metrics
            monitor.record_inference(inference_time)
            monitor.record_system_stats()

    return monitor
```

### 2. Analysis & Reporting

```python
def analyze_performance(monitor):
    times = np.array(monitor.inference_times)
    memory = np.array(monitor.memory_usage)
    temp = np.array(monitor.temperature)

    report = {
        'inference': {
            'mean': np.mean(times),
            'std': np.std(times),
            'p95': np.percentile(times, 95)
        },
        'memory': {
            'mean': np.mean(memory),
            'max': np.max(memory)
        },
        'temperature': {
            'mean': np.mean(temp),
            'max': np.max(temp)
        }
    }

    return report
```

## Best Practices

1. **Memory Management**

   - Preallocate arrays where possible
   - Use grayscale images to reduce memory
   - Monitor and log memory usage

2. **Inference Optimization**

   - Use quantized models
   - Optimize thread count
   - Enable graph optimizations

3. **System Configuration**

   - Set appropriate CPU governor
   - Configure adequate swap space
   - Monitor temperature

4. **Monitoring**
   - Log performance metrics
   - Track resource usage
   - Monitor system temperature
