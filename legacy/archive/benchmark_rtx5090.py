#!/usr/bin/env python3
"""
RTX 5090 Performance Benchmark for Cutlery Classifier
Tests inference speed to verify 50ms requirement
"""

import time
import torch
import statistics
from scripts.run_inference import load_model, preprocess_image, get_predictions


def benchmark_rtx5090():
    """Benchmark inference performance on RTX 5090"""
    print("🚀 RTX 5090 PERFORMANCE BENCHMARK")
    print("=" * 50)

    # Setup
    device = torch.device("cuda")
    print(f"📱 Device: {torch.cuda.get_device_name(0)}")
    print(
        f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Load model and measure load time
    print("\n📦 Loading model...")
    start_load = time.time()
    model, class_names = load_model("models/checkpoints/type_detector_best.pth", device)
    load_time = time.time() - start_load
    print(f"⏱️ Model Load Time: {load_time * 1000:.1f}ms")

    # Prepare test image
    image_path = "data/raw/fork/IMG_0941[1].jpg"
    image_tensor = preprocess_image(image_path)
    print(f"🖼️ Test Image: {image_path}")

    # GPU Warmup
    print("\n🔥 GPU Warmup (5 iterations)...")
    for _ in range(5):
        _ = get_predictions(model, image_tensor, class_names, device)

    # Benchmark inference
    print("\n⚡ Running Benchmark (100 iterations)...")
    times = []
    for i in range(100):
        start = time.time()
        predictions = get_predictions(model, image_tensor, class_names, device)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        if (i + 1) % 25 == 0:
            print(f"   Progress: {i + 1}/100")

    # Calculate statistics
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times)

    # Results
    print("\n📊 PERFORMANCE RESULTS")
    print("-" * 30)
    print(f"Average Time: {avg_time:.2f}ms")
    print(f"Median Time:  {median_time:.2f}ms")
    print(f"Min Time:     {min_time:.2f}ms")
    print(f"Max Time:     {max_time:.2f}ms")
    print(f"Std Dev:      {std_time:.2f}ms")

    # Requirement check
    req_check = "PASS" if avg_time < 50 else "FAIL"
    print(f"\n🎯 50ms Requirement: {req_check} ({avg_time:.1f}ms < 50ms)")

    # Prediction result
    predicted_class = max(predictions, key=predictions.get)
    confidence = max(predictions.values())
    print(f"🔍 Prediction: {predicted_class} ({confidence:.3f})")

    # Performance classification
    if avg_time < 25:
        performance = "EXCELLENT"
    elif avg_time < 35:
        performance = "VERY GOOD"
    elif avg_time < 50:
        performance = "GOOD"
    else:
        performance = "NEEDS OPTIMIZATION"

    print(f"⭐ Performance Rating: {performance}")

    return {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "passes_50ms_req": avg_time < 50,
        "performance_rating": performance,
    }


if __name__ == "__main__":
    results = benchmark_rtx5090()
