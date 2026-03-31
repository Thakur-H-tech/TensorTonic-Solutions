import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    
    # Determine max length
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)
    
    # Create output array filled with pad_value
    padded = np.full((N, max_len), pad_value)
    
    # Fill values
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    
    return padded
    pass