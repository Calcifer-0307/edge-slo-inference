import onnxruntime as ort
import numpy as np
import pandas as pd
import time
import os

def profile_model(model_path: str, model_name: str, warmup_runs: int = 20, profile_runs: int = 200) -> dict:
    print(f"Profiling {model_name}...")
    
    # Use CPU Execution Provider
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = os.cpu_count() or 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    # Dummy input based on standard shape: (batch_size=1, channels=3, H=224, W=224)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warm-up phase
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})
        
    # Profiling phase
    latencies = []
    for _ in range(profile_runs):
        start_time = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end_time = time.perf_counter()
        
        # Convert seconds to milliseconds
        latencies.append((end_time - start_time) * 1000)
        
    latencies = np.array(latencies)
    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    p50_ms = np.percentile(latencies, 50)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)
    
    print(f"  {model_name}: Mean = {mean_ms:.2f} ms, Std = {std_ms:.2f} ms, P99 = {p99_ms:.2f} ms")
    
    return {
        'name': model_name,
        'service_time_mean_ms': mean_ms,
        'service_time_std_ms': std_ms,
        'service_time_p50_ms': p50_ms,
        'service_time_p95_ms': p95_ms,
        'service_time_p99_ms': p99_ms
    }

def main():
    model_dir = "profiling/models"
    models_to_profile = [
        {"name": "fast", "file": "mobilenet_v2.onnx"},
        {"name": "mid", "file": "resnet18.onnx"},
        {"name": "slow", "file": "resnet50.onnx"}
    ]

    missing_models = [m["file"] for m in models_to_profile if not os.path.exists(os.path.join(model_dir, m["file"]))]
    if missing_models:
        print("Missing local ONNX models:")
        for file_name in missing_models:
            print(f"  - {os.path.join(model_dir, file_name)}")
        print("\nRun `python3 profiling/generate_onnx.py` first to download/export the local profiling models.")
        return

    results = []
    for m in models_to_profile:
        model_path = os.path.join(model_dir, m['file'])
        stats = profile_model(model_path, m['name'])
        results.append(stats)
            
    df = pd.DataFrame(results)
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/service_time.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nProfiling completed. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
