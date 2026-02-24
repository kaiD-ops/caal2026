import torch.nn as nn

class TakeLastTimestep(nn.Module):
    """
    Module that extracts the last timestep from a sequence.

    This layer is used to summarize sequence outputs from recurrent 
    or sequence models by taking only the final timestep as a feature vector.

    Parameters
    ----------
    None

    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, L, D), where
        B : batch size,
        L : sequence length,
        D : feature dimension.

    Returns
    -------
    out : torch.Tensor
        Output tensor of shape (B, D), corresponding to the last timestep
        of each sequence in the batch.
    """
    def forward(self, x):
        """
        Extract the last timestep from a sequence tensor.
        
        The final hidden state x_L encodes information about the entire sequence
        through recurrent dynamics, making it suitable for sequence classification.
        This is analogous to using the final hidden state in RNN architectures.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D)
        
        Returns
        -------
        torch.Tensor
            Last timestep tensor of shape (B, D)
        """
        # Extract last timestep: input[:, -1, :] selects the final position
        # in the sequence dimension (dimension 1)
        return x[:, -1, :]