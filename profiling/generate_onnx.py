import torch
import torchvision.models as models
import os
import ssl
import urllib.request

# Bypass SSL certificate verification locally to download models
ssl._create_default_https_context = ssl._create_unverified_context

def export_model_to_onnx(model, model_name: str, dummy_input: torch.Tensor, output_dir: str):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    print(f"Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported to {output_path}")

def main():
    # Use standard ImageNet input shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    output_dir = "profiling/models"
    
    # 1. Fast model (MobileNetV2) - Lower accuracy proxy
    fast_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    export_model_to_onnx(fast_model, "mobilenet_v2", dummy_input, output_dir)
    
    # 2. Mid model (ResNet18) - Medium accuracy proxy
    mid_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    export_model_to_onnx(mid_model, "resnet18", dummy_input, output_dir)
    
    # 3. Slow model (ResNet50) - Higher accuracy proxy
    slow_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    export_model_to_onnx(slow_model, "resnet50", dummy_input, output_dir)

if __name__ == "__main__":
    main()
