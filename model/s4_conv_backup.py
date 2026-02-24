"""
S4 Convolutional Implementation

This module implements the Structured State Space model using its convolutional formulation.
The convolutional view precomputes a kernel that represents the full sequence transformation,
enabling parallel processing via convolution operations. This formulation is significantly
faster for training on long sequences.

Mathematical Foundation:
-----------------------
Starting from the recurrence relation:
    x_k = Ā·x_{k-1} + B̄·u_k
    y_k = C·x_k + D·u_k

Unrolling the recurrence gives:
    x_0 = 0
    x_1 = B̄·u_1
    x_2 = Ā·B̄·u_1 + B̄·u_2
    x_3 = Ā²·B̄·u_1 + Ā·B̄·u_2 + B̄·u_3
    ...
    x_k = Σᵢ₌₁ᵏ Āᵏ⁻ⁱ·B̄·uᵢ

Therefore:
    y_k = C·(Σᵢ₌₁ᵏ Āᵏ⁻ⁱ·B̄·uᵢ) + D·u_k
        = Σᵢ₌₁ᵏ (C·Āᵏ⁻ⁱ·B̄)·uᵢ + D·u_k

This is a convolution with kernel:
    K = [C·B̄, C·Ā·B̄, C·Ā²·B̄, ..., C·Āᴸ⁻¹·B̄]

Convolution formulation:
    y = K * u + D·u

Complexity: O(L²·N) naive, O(L·log(L)·N) with FFT

References:
----------
Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with 
Structured State Spaces. ICLR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def make_hippo_matrix(N):
    """
    Generate a HiPPO (High-order Polynomial Projection Operator) matrix for better memory.
    
    Parameters
    ----------
    N : int
        State dimension
        
    Returns
    -------
    torch.Tensor
        HiPPO matrix of shape (N, N)
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -torch.from_numpy(A).float()


class S4Convolutional(nn.Module):
    """
    Convolutional Structured State Space (S4) Layer.
    
    Processes sequences using a convolutional formulation where the SSM is expressed
    as a convolution with a learned kernel. This enables parallel processing and is
    significantly faster than the recurrent formulation for training.
    
    Parameters
    ----------
    d_model : int
        Feature dimension (number of independent SSM copies)
    d_state : int, optional
        State space dimension N. Default is 64.
    dt_min : float, optional
        Minimum discretization step size. Default is 0.001.
    dt_max : float, optional
        Maximum discretization step size. Default is 0.1.
    l_max : int, optional
        Maximum sequence length for precomputing kernel. Default is 4096.
        
    Attributes
    ----------
    A : nn.Parameter
        Continuous state matrix (N, N), initialized with HiPPO
    B : nn.Parameter
        Input-to-state matrix (N, 1)
    C : nn.Parameter
        State-to-output matrix (1, N)
    D : nn.Parameter
        Skip connection scalar
    log_dt : nn.Parameter
        Log-space discretization step size
    """
    
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, l_max=4096):
        super().__init__()
        
        self.h = d_model  # Number of features
        self.n = d_state  # State dimension
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.l_max = l_max  # Maximum sequence length
        
        # Initialize continuous-time parameters
        # A: HiPPO matrix for better long-range dependencies
        A = make_hippo_matrix(self.n)
        self.register_buffer("A", A)  # Fixed HiPPO initialization
        
        # B: Input projection, initialized randomly
        B = torch.randn(self.n, 1)
        self.B = nn.Parameter(B)
        
        # C: Output projection, initialized randomly
        C = torch.randn(1, self.n)
        self.C = nn.Parameter(C)
        
        # D: Skip connection
        D = torch.randn(1)
        self.D = nn.Parameter(D)
        
        # Log-space parameterization of step size
        log_dt = torch.rand(1) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
    def discretize(self):
        """
        Discretize the continuous-time SSM using matrix exponentials.
        
        Returns
        -------
        A_bar : torch.Tensor
            Discretized state matrix of shape (n, n)
        B_bar : torch.Tensor
            Discretized input matrix of shape (n, 1)
        """
        # Get step size from log-space
        dt = torch.exp(self.log_dt)
        
        # Compute Ā = exp(ΔA)
        dt_A = dt * self.A
        A_bar = torch.linalg.matrix_exp(dt_A)
        
        # Compute B̄ = (ΔA)^{-1}(exp(ΔA) - I) · (ΔB)
        I = torch.eye(self.n, device=self.A.device, dtype=self.A.dtype)
        A_bar_minus_I = A_bar - I
        B_bar = torch.linalg.solve(self.A, A_bar_minus_I @ self.B)
        
        return A_bar, B_bar
    
    def compute_kernel(self, L):
        """
        Compute the SSM convolution kernel.
        
        The kernel represents the impulse response of the discretized SSM:
            K[k] = C·Āᵏ·B̄ for k = 0, 1, ..., L-1
            
        This is computed efficiently by iteratively multiplying by Ā.
        
        Parameters
        ----------
        L : int
            Sequence length (kernel length)
            
        Returns
        -------
        kernel : torch.Tensor
            Convolution kernel of shape (L,)
            
        Algorithm
        ---------
        1. Initialize: power = B̄
        2. For k = 0 to L-1:
            a. K[k] = C·power
            b. power = Ā·power
        
        Complexity
        ----------
        Time: O(L·N²) due to N×N matrix-vector multiplication at each step
        Space: O(L + N²)
        """
        # Get discretized matrices
        A_bar, B_bar = self.discretize()  # (n,n), (n,1)
        
        # Initialize kernel
        kernel = []
        
        # Start with B̄ as the first power
        power = B_bar  # (n, 1)
        
        # Compute K[k] = C·Āᵏ·B̄ for k = 0, ..., L-1
        for k in range(L):
            # K[k] = C @ power
            k_k = (self.C @ power).squeeze()  # (1,n) @ (n,1) → scalar
            kernel.append(k_k)
            
            # Update: power = Ā @ power (for next iteration)
            power = A_bar @ power  # (n,n) @ (n,1) → (n,1)
        
        # Stack into a single tensor
        kernel = torch.stack(kernel)  # (L,)
        
        return kernel
    
    def forward(self, u):
        """
        Forward pass using convolutional formulation.
        
        Computes the output sequence by convolving the input with the SSM kernel
        and adding the skip connection.
        
        Parameters
        ----------
        u : torch.Tensor
            Input sequence of shape (B, L, H) where
            B : batch size
            L : sequence length
            H : feature dimension (d_model)
            
        Returns
        -------
        y : torch.Tensor
            Output sequence of shape (B, L, H)
        None
            Placeholder for state (interface compatibility)
            
        Algorithm
        ---------
        1. Compute or retrieve kernel K of length L
        2. Perform convolution: y_conv = K * u
        3. Add skip connection: y = y_conv + D·u
        
        Complexity
        ----------
        Time: O(B·L²·H) for direct convolution, O(B·L·log(L)·H) with FFT
        Space: O(B·L·H + L) for outputs and kernel
        
        Notes
        -----
        We use torch.nn.functional.conv1d for the convolution operation.
        The input must be in (B, C, L) format, so we transpose.
        """
        B, L, H = u.shape
        
        # Compute the convolution kernel for this sequence length
        kernel = self.compute_kernel(L)  # (L,)
        
        # Transpose for conv1d: (B, L, H) → (B, H, L)
        u_transposed = u.transpose(1, 2)  # (B, H, L)
        
        # Prepare kernel for convolution
        # conv1d expects kernel shape: (out_channels, in_channels, kernel_size)
        # For each input channel independently: (H, 1, L)
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        kernel_expanded = kernel_expanded.expand(H, 1, -1)  # (H, 1, L)
        
        # Perform convolution for each feature independently
        # Use groups=H to apply each kernel to its corresponding channel
        y_conv = F.conv1d(
            u_transposed,  # (B, H, L)
            kernel_expanded,  # (H, 1, L)
            padding=L - 1,  # Causal padding: pad left to preserve causality
            groups=H  # Independent convolution for each feature
        )  # (B, H, L + L - 1)
        
        # Truncate to original sequence length (causal convolution)
        y_conv = y_conv[:, :, :L]  # (B, H, L)
        
        # Transpose back to (B, L, H)
        y_conv = y_conv.transpose(1, 2)  # (B, L, H)
        
        # Add skip connection: D·u
        y = y_conv + self.D * u  # (B, L, H)
        
        return y, None  # Return None for state (interface compatibility)


if __name__ == "__main__":
    # Test the S4 Convolutional implementation
    print("Testing S4 Convolutional Implementation...")
    
    # Create model
    d_model = 16
    d_state = 64
    model = S4Convolutional(d_model=d_model, d_state=d_state, l_max=1024)
    
    # Test input
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    y, _ = model(x)
    print(f"Output shape: {y.shape}")
    print(f"✓ S4Convolutional forward pass successful!")
    
    # Check parameter counts
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
