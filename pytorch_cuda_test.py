import torch

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        try:
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Could not get device name: {e}")
    else:
        print("CUDA version: None")
        print("Device name: None") 