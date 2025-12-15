
"""Determinism utilities for reproducible experiments.

This module provides utilities for setting up deterministic behavior
across PyTorch, NumPy, and CUDA to ensure reproducible experiments.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def setup_determinism(
    enable_cuda_determinism: bool = True,
    cublas_workspace_config: str = ":4096:8",
    set_num_threads: int = 1,
    verbose: bool = True
) -> None:
    """Set up deterministic behavior for reproducible experiments.
    
    Args:
        enable_cuda_determinism: Whether to enable CUDA deterministic algorithms
        cublas_workspace_config: CUBLAS workspace configuration for determinism
        set_num_threads: Number of PyTorch threads (1 for determinism)
        verbose: Whether to print status messages
    """
    if verbose:
        print("🔧 Setting up deterministic environment...")
    
    # Set PyTorch thread count for reproducibility
    torch.set_num_threads(set_num_threads)
    if verbose:
        print(f"   → PyTorch threads: {set_num_threads}")
    
    # Python and CUDA environment settings
    os.environ["PYTHONUNBUFFERED"] = "1"
    if verbose:
        print("   → PYTHONUNBUFFERED: 1 (immediate output)")
    
    # CUDA reproducibility settings
    if cublas_workspace_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        if verbose:
            print(f"   → CUBLAS_WORKSPACE_CONFIG: {cublas_workspace_config}")
    
    # Enable full CUDA determinism if available
    if torch.cuda.is_available() and enable_cuda_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Critical: Disable TF32 for full bit-identical determinism across A100/V100
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        if verbose:
            print("   → cuDNN deterministic: True")
            print("   → cuDNN benchmark: False")
            print("   → TF32 matmul: Disabled")
            print("   → TF32 cuDNN: Disabled")
        
        try:
            torch.use_deterministic_algorithms(True)
            if verbose:
                print("   → PyTorch deterministic algorithms: Enabled")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not enable deterministic algorithms: {e}")
                print("      Proceeding with other determinism settings...")
    elif verbose:
        if not torch.cuda.is_available():
            print("   → CUDA not available, skipping CUDA determinism")
        else:
            print("   → CUDA determinism disabled by user")
    
    if verbose:
        print("✅ Deterministic environment setup completed")


def set_experiment_seeds(
    seed: int,
    include_cuda: bool = True,
    verbose: bool = False
) -> None:
    """Set all random seeds for reproducible experiments.
    
    This function sets seeds for all major random number generators:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generator
    - CUDA random number generators (if available)
    
    Args:
        seed: Random seed to use
        include_cuda: Whether to set CUDA seeds (if available)
        verbose: Whether to print seed information
    """
    if verbose:
        print(f"🌱 Setting experiment seeds to {seed}")
    
    # Set Python's built-in random seed (for random.sample, etc.)
    random.seed(seed)
    # Ensure Python hashing is deterministic across runs and processes
    os.environ["PYTHONHASHSEED"] = str(seed)
    if verbose:
        print(f"   → Python random seed: {seed}")
        print(f"   → PYTHONHASHSEED: {os.environ['PYTHONHASHSEED']}")
    
    # Set NumPy seed
    np.random.seed(seed)
    if verbose:
        print(f"   → NumPy seed: {seed}")
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    if verbose:
        print(f"   → PyTorch seed: {seed}")
    
    # Set CUDA seeds if available and requested
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"   → CUDA seed: {seed}")
            print(f"   → CUDA all devices seed: {seed}")
    elif verbose and not torch.cuda.is_available():
        print("   → CUDA not available, skipping CUDA seeds")


def ensure_gpu_determinism(verbose: bool = True) -> None:
    """Ensure GPU operations are deterministic and clean up state.
    
    Args:
        verbose: Whether to print status information
    """
    if not torch.cuda.is_available():
        if verbose:
            print("   → CUDA not available, skipping GPU determinism checks")
        return
    
    if verbose:
        print("🔍 Ensuring GPU determinism...")
    
    # Force single GPU usage for consistency
    torch.cuda.set_device(0)
    if verbose:
        print("   → Using GPU device 0")
    
    # Clear GPU cache completely
    torch.cuda.empty_cache()
    if verbose:
        print("   → GPU cache cleared")
    
    # Synchronize before starting
    torch.cuda.synchronize()
    if verbose:
        print("   → GPU synchronized")
    
    # Check GPU memory status
    if verbose:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        print(f"   → GPU Memory: {gpu_memory/1e9:.1f}GB total, {allocated/1e9:.1f}GB allocated")
    
    # Additional determinism for shared environments
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if verbose:
        print("   → CUDA_LAUNCH_BLOCKING: 1 (synchronous execution)")


def create_dataloader_determinism(seed: int):
    """Create components to make PyTorch DataLoader behavior deterministic.
    
    Returns a tuple (generator, seed_worker) to be passed to DataLoader as
    `generator=...` and `worker_init_fn=...` when using multiple workers.
    If you use `num_workers=0`, only `generator` is typically needed for
    deterministic shuffling.
    """
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g, seed_worker


def setup_full_determinism(
    seed: int,
    enable_cuda_determinism: bool = True,
    ensure_gpu_state: bool = True,
    verbose: bool = True
) -> None:
    """Complete determinism setup with seed setting.
    
    This is a convenience function that combines setup_determinism(),
    set_experiment_seeds(), and optionally ensure_gpu_determinism().
    
    Args:
        seed: Random seed to use for all generators
        enable_cuda_determinism: Whether to enable CUDA deterministic algorithms
        ensure_gpu_state: Whether to ensure clean GPU state
        verbose: Whether to print detailed status information
    """
    if verbose:
        print("🎯 Setting up complete deterministic environment...")
        print("=" * 60)
    
    # Step 1: Setup deterministic algorithms and environment
    setup_determinism(
        enable_cuda_determinism=enable_cuda_determinism,
        verbose=verbose
    )
    
    # Step 2: Set all random seeds
    set_experiment_seeds(
        seed=seed,
        include_cuda=enable_cuda_determinism,
        verbose=verbose
    )
    
    # Step 3: Ensure GPU state if requested
    if ensure_gpu_state:
        ensure_gpu_determinism(verbose=verbose)
    
    if verbose:
        print("=" * 60)
        print(f"✅ Complete deterministic setup completed with seed {seed}")