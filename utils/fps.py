import time
import torch
from tqdm import tqdm
from dagr.utils.buffers import format_data

def measure_fps(model, test_loader, device, warmup_batches=5, num_batches=20):
    """
    Measure model inference FPS (frames per second)
    
    Args:
        model: Model to test
        test_loader: Test data loader
        device: Computation device (CPU or GPU)
        warmup_batches: Number of warmup batches, ignore timing for these to avoid initial delay effects
        num_batches: Number of batches used for FPS calculation
        
    Returns:
        dict: Dictionary containing FPS and related measurement data
    """
    print("\nStarting FPS measurement...")
    
    # Variables for recording time and batch information
    batch_times = []
    batch_sizes = []
    
    # Ensure model is in evaluation mode
    model.eval()

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Ensure CUDA operations are synchronized
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="FPS Measurement Progress", total=warmup_batches+num_batches)):
            # Skip invalid batches
            if not hasattr(data, 'bbox') or data.bbox is None or (isinstance(data.bbox, torch.Tensor) and data.bbox.shape[0] == 0):
                continue
                
            # Move data to device and prepare
            data = data.to(device)
            data = format_data(data)
            labels = data.bbox[:, 4]
            
            # Skip warmup batches
            if i < warmup_batches:
                # Perform inference but don't record time
                _ = model(data, labels, testing=True)
                continue
                
            # Stop if enough batches have been collected
            if len(batch_times) >= num_batches:
                break
            
            # Record batch size
            batch_size = len(data.bbox) + len(data.bbox0) if hasattr(data, 'bbox') else 0
            batch_sizes.append(batch_size)
            
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            start_time = time.time()
            
            # Forward pass
            _ = model(data, labels, testing=True)
            
            # Ensure inference is complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Record batch processing time
            batch_time = end_time - start_time
            batch_times.append(batch_time)
    
    
    # Calculate statistics
    total_time = sum(batch_times)
    total_samples = sum(batch_sizes)
    avg_batch_size = total_samples / len(batch_sizes) if batch_sizes else 0
    
    # Calculate FPS
    fps = total_samples / total_time if total_time > 0 else 0
    batch_fps = len(batch_times) / total_time if total_time > 0 else 0
    
    print(f"\nFPS Measurement Results:")
    print(f"Average frames per batch: {avg_batch_size:.2f}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"FPS: {fps:.2f} ")
    
    return {
        'fps': fps,
        'avg_batch_size': avg_batch_size,
        'total_time': total_time
    }