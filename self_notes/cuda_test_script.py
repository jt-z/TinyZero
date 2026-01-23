import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
print("-" * 30)

if not torch.cuda.is_available():
    print("CUDA is not available to PyTorch.")
else:
    print("CUDA is available!")

    try:
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s).")

        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        print("\nTesting tensor operations on cuda:0...")
        # Create a tensor and move it to the GPU
        x = torch.tensor([1.0, 2.0]).to("cuda:0")
        print("Tensor created on CPU: ", torch.tensor([1.0, 2.0]))
        print("Tensor successfully moved to cuda:0: ", x)

        # Perform a simple operation
        y = x * 2
        print("Simple operation (tensor * 2) on GPU successful: ", y)
        print("\nBasic CUDA functionality test successful!")

    except Exception as e:
        print("\nAn error occurred during CUDA test:")
        print(e)
