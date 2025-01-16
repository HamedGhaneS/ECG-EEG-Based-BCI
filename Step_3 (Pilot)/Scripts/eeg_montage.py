"""
Standard 64-Channel EEG Montage Definition for BrainCap MR 64
https://www.brainproducts.com/downloads/cap-montages/

Following the extended 10-20 system with additional positions

This montage file provides a standardized reference for EEG channel locations
and groupings commonly used in cognitive neuroscience research. It includes:
1. Channel names and indices
2. Regional channel groupings
3. Hemisphere groupings
4. Standardized coordinate system references
"""

import numpy as np

# Basic channel information
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
    'AF1', 'AF2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO1', 'PO2', 'FC5', 'FC6',
    'CP5', 'CP6', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF5', 'AF6',
    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6',
    'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
    'Fpz', 'FCz', 'CPz', 'ECG'  # ECG is the 64th channel
]

# Channel indices (0-based indexing)
CHANNEL_INDICES = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

# Regional channel groupings
CHANNEL_GROUPS = {
    'Frontal': [
        'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'AF1', 'AF2',
        'F1', 'F2', 'AF5', 'AF6', 'F5', 'F6', 'AF7', 'AF8', 'Fpz'
    ],
    'Central': [
        'C3', 'C4', 'Cz', 'C1', 'C2', 'C5', 'C6'
    ],
    'Temporal': [
        'T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8'
    ],
    'Parietal': [
        'P3', 'P4', 'P7', 'P8', 'Pz', 'P1', 'P2', 'P5', 'P6'
    ],
    'Occipital': [
        'O1', 'O2', 'Oz', 'PO1', 'PO2', 'PO3', 'PO4', 'PO7', 'PO8'
    ],
    'Frontocentral': [
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz'
    ],
    'Centroparietal': [
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz'
    ]
}

# Hemisphere groupings
HEMISPHERE_GROUPS = {
    'Left': [ch for ch in CHANNEL_NAMES if ch[-1].isdigit() and int(ch[-1]) % 2 != 0],
    'Midline': [ch for ch in CHANNEL_NAMES if ch.endswith('z')],
    'Right': [ch for ch in CHANNEL_NAMES if ch[-1].isdigit() and int(ch[-1]) % 2 == 0]
}

# Common analysis groups
ANALYSIS_GROUPS = {
    'Frontocentral': ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'C3', 'Cz', 'C4'],
    'Centroparietal': ['C3', 'Cz', 'C4', 'CP1', 'CP2', 'P3', 'Pz', 'P4'],
    'Frontotemporal': ['F7', 'F8', 'FT7', 'FT8', 'T7', 'T8'],
    'Parietooccipital': ['P7', 'P8', 'PO7', 'PO8', 'O1', 'Oz', 'O2']
}

def get_channel_indices(channel_names):
    """
    Convert channel names to their corresponding indices.
    
    Args:
        channel_names (list): List of channel names
        
    Returns:
        list: List of corresponding channel indices (0-based)
    """
    return [CHANNEL_INDICES[ch] for ch in channel_names]

def get_group_indices(group_name):
    """
    Get channel indices for a predefined channel group.
    
    Args:
        group_name (str): Name of the channel group
        
    Returns:
        list: Channel indices for the specified group
    """
    if group_name in CHANNEL_GROUPS:
        return get_channel_indices(CHANNEL_GROUPS[group_name])
    elif group_name in ANALYSIS_GROUPS:
        return get_channel_indices(ANALYSIS_GROUPS[group_name])
    else:
        raise ValueError(f"Unknown group name: {group_name}")

def get_hemisphere_indices(hemisphere):
    """
    Get channel indices for a specific hemisphere.
    
    Args:
        hemisphere (str): 'Left', 'Right', or 'Midline'
        
    Returns:
        list: Channel indices for the specified hemisphere
    """
    if hemisphere in HEMISPHERE_GROUPS:
        return get_channel_indices(HEMISPHERE_GROUPS[hemisphere])
    else:
        raise ValueError("Hemisphere must be 'Left', 'Right', or 'Midline'")

# Example usage:
if __name__ == "__main__":
    # Print channel information
    print("Total number of channels:", len(CHANNEL_NAMES))
    print("\nChannel indices example:")
    print("Fz index:", CHANNEL_INDICES['Fz'])
    print("Cz index:", CHANNEL_INDICES['Cz'])
    
    # Print group information
    print("\nFrontocentral channels:", ANALYSIS_GROUPS['Frontocentral'])
    print("Frontocentral indices:", get_group_indices('Frontocentral'))
    
    # Print hemisphere information
    print("\nMidline channels:", HEMISPHERE_GROUPS['Midline'])
    print("Left hemisphere indices:", get_hemisphere_indices('Left'))
