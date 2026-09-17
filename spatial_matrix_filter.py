import numpy as np

def apply_3x3_convolution_filter(data_grid):
    """
    Applies a discrete 3x3 Hanning convolution matrix to isolate physical target signatures
    from background system noise drift across a continuous spatial grid array.
    """
    # Step 1: Define the 3x3 convolution kernel coefficients (Hanning weights)
    hanning_kernel = np.array([
        [0.0625, 0.125,  0.0625],
        [0.125,  0.25,   0.125 ],
        [0.0625, 0.125,  0.0625]
    ])
    
    # Get spatial grid layout boundaries
    rows, cols = data_grid.shape
    output_grid = np.copy(data_grid)
    
    # Step 2: Iterate across the discrete 2D spatial coordinate system
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Extract the local 3x3 sub-matrix tracking localized hardware drift
            local_sub_grid = data_grid[r-1:r+2, c-1:c+2]
            
            # Execute element-wise matrix multiplication and compute the scalar sum
            output_grid[r, c] = np.sum(local_sub_grid * hanning_kernel)
            
    return output_grid

# ==========================================
# SIMULATION VALIDATION LOOP
# ==========================================
if __name__ == "__main__":
    # Create a dummy 5x5 spatial data grid simulating background hardware telemetry noise
    np.random.seed(42)
    raw_telemetry_grid = np.random.uniform(low=10.0, high=12.0, size=(5, 5))
    
    print("Raw Telemetry Grid (Before Noise Suppression Inversion):")
    print(np.round(raw_telemetry_grid, 3))
    print("\n" + "="*50 + "\n")
    
    # Process the grid through our custom 3x3 numeric inversion loop
    cleaned_signal_matrix = apply_3x3_convolution_filter(raw_telemetry_grid)
    
    print("Cleaned Target Grid (After 3x3 Discrete Spatial Filtering):")
    print(np.round(cleaned_signal_matrix, 3))
