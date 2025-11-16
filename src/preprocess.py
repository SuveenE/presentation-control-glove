"""
IMU Data Preprocessing Module

This module extracts 84 features from 2-IMU sensor data (wrist + index finger).
It contains two main sections:

1. CORE PREPROCESSING (Lines 20-180) - Required for live/streaming applications
   - Helper functions
   - Feature extraction functions
   - Main API: features_from_12xN()
   
2. BATCH PROCESSING UTILITIES (Lines 182+) - For offline batch processing
   - Directory processing
   - Command-line interface
   
For live streaming applications, only copy the CORE PREPROCESSING section.
"""

# Core imports (required for live streaming)
from typing import Dict, Sequence, Union
import pandas as pd
import numpy as np
import time

# Batch processing imports (optional - only for offline processing)
from typing import Optional
import json
import argparse
from pathlib import Path
try:
    import joblib
except ImportError:
    joblib = None
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None

# =============================================================================
# CORE PREPROCESSING - START
# Copy everything from here to "CORE PREPROCESSING - END" for live streaming
# =============================================================================

DEFAULT_COL_ORDER: Sequence[str] = [
    # IMU0 (wrist)
    'Imu0_linear_accleration_x','Imu0_linear_accleration_y','Imu0_linear_accleration_z',
    'Imu0_angular_velocity_x','Imu0_angular_velocity_y','Imu0_angular_velocity_z',
    # IMU1 (index finger)
    'Imu1_linear_accleration_x','Imu1_linear_accleration_y','Imu1_linear_accleration_z',
    'Imu1_angular_velocity_x','Imu1_angular_velocity_y','Imu1_angular_velocity_z',
]

# ---------- helpers ----------
def vec_mag(x, y, z):
    return np.sqrt(np.asarray(x)**2 + np.asarray(y)**2 + np.asarray(z)**2)

def stats_5(arr):
    """Compute mean, std, rms, max, median for an array."""
    a = np.asarray(arr)
    return {
        'mean': float(np.mean(a)),
        'std':  float(np.std(a, ddof=0)),
        'rms':  float(np.sqrt(np.mean(a**2))),
        'max':  float(np.max(a)),
        'median': float(np.median(a)),
    }

def series_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# ---------- per-IMU feature blocks ----------
def imu_40_features(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """
    Extract 40 features for a single IMU (wrist or index finger):
      - Axis stats: 5 stats × 3 axes × 2 signals (acc, gyro) = 30 features
        Stats: mean, std, rms, max, median
      - Magnitude stats: 5 stats × 2 magnitudes (accmag, gyromag) = 10 features
        Stats: mean, std, rms, max, median
    Total: 40 features
    """
    feats = {}

    # Axis stats (30 features: 5 stats × 3 axes × 2 signals)
    for sig, sig_name in [('linear_accleration', 'acc'), ('angular_velocity', 'gyro')]:
        for ax in ['x', 'y', 'z']:
            s = stats_5(df[f'{prefix}_{sig}_{ax}'])
            feats[f'{prefix}_{sig_name}_{ax}_mean'] = s['mean']
            feats[f'{prefix}_{sig_name}_{ax}_std'] = s['std']
            feats[f'{prefix}_{sig_name}_{ax}_rms'] = s['rms']
            feats[f'{prefix}_{sig_name}_{ax}_max'] = s['max']
            feats[f'{prefix}_{sig_name}_{ax}_median'] = s['median']

    # Magnitude stats (10 features: 5 stats × 2 magnitudes)
    acc_mag = vec_mag(df[f'{prefix}_linear_accleration_x'],
                      df[f'{prefix}_linear_accleration_y'],
                      df[f'{prefix}_linear_accleration_z'])
    gyro_mag = vec_mag(df[f'{prefix}_angular_velocity_x'],
                       df[f'{prefix}_angular_velocity_y'],
                       df[f'{prefix}_angular_velocity_z'])

    for name, arr in [('accmag', acc_mag), ('gyromag', gyro_mag)]:
        s = stats_5(arr)
        feats[f'{prefix}_{name}_mean'] = s['mean']
        feats[f'{prefix}_{name}_std'] = s['std']
        feats[f'{prefix}_{name}_rms'] = s['rms']
        feats[f'{prefix}_{name}_max'] = s['max']
        feats[f'{prefix}_{name}_median'] = s['median']

    return feats  # 40

# ---------- cross-IMU (4 features) ----------
def cross_4(df: pd.DataFrame) -> Dict[str, float]:
    """
    Cross-IMU features between wrist (W) and finger (F):
      - 2 correlations (acc & gyro magnitude time series)
      - 2 RMS ratios (acc & gyro, finger over wrist)
    Total: 4 features
    """
    feats = {}

    # Magnitudes for wrist (W) and finger (F)
    W_acc = vec_mag(df['Imu0_linear_accleration_x'], 
                    df['Imu0_linear_accleration_y'], 
                    df['Imu0_linear_accleration_z'])
    F_acc = vec_mag(df['Imu1_linear_accleration_x'], 
                    df['Imu1_linear_accleration_y'], 
                    df['Imu1_linear_accleration_z'])

    W_gyr = vec_mag(df['Imu0_angular_velocity_x'], 
                    df['Imu0_angular_velocity_y'], 
                    df['Imu0_angular_velocity_z'])
    F_gyr = vec_mag(df['Imu1_angular_velocity_x'], 
                    df['Imu1_angular_velocity_y'], 
                    df['Imu1_angular_velocity_z'])

    # 2 correlations
    feats['corr_acc_W_F'] = series_corr(W_acc, F_acc)
    feats['corr_gyr_W_F'] = series_corr(W_gyr, F_gyr)

    # 2 RMS ratios (finger over wrist)
    eps = 1e-9
    feats['ratio_acc_rms_F_over_W'] = float(np.sqrt(np.mean(F_acc**2)) / (np.sqrt(np.mean(W_acc**2)) + eps))
    feats['ratio_gyr_rms_F_over_W'] = float(np.sqrt(np.mean(F_gyr**2)) / (np.sqrt(np.mean(W_gyr**2)) + eps))

    return feats  # 4

# ---------- main: 84 features ----------
def extract_84_features(df_window: pd.DataFrame) -> Dict[str, float]:
    """
    Given a 3s window (DataFrame), compute the 84-d feature vector using ONLY accel+gyro.
      IMU0 (wrist): 40 features
      IMU1 (index finger): 40 features
      Cross-IMU: 4 features
    Total = 84
    """
    # required columns (only accel + gyro)
    missing = [c for c in DEFAULT_COL_ORDER if c not in df_window.columns]
    if missing:
        raise ValueError(f"Missing columns in window data: {missing[:5]}{'...' if len(missing)>5 else ''}")

    feats = {}
    feats.update(imu_40_features(df_window, 'Imu0'))  # 40
    feats.update(imu_40_features(df_window, 'Imu1'))  # 40
    feats.update(cross_4(df_window))                   # 4

    assert len(feats) == 84, f"Expected 84 features, got {len(feats)}"
    return feats


def features_from_12xN(
    data: Union[pd.DataFrame, np.ndarray],
    col_order: Sequence[str] = DEFAULT_COL_ORDER,
    timing: bool = False
) -> Dict[str, float]:
    """
    Extract 84 features from 2-IMU sensor data (12 columns: 2 IMUs × 6 measurements).
    
    *** MAIN API FOR LIVE STREAMING ***
    This is the primary function to call when processing streaming IMU data.
    
    Args:
        data: Either a numpy array of shape (N, 12) or a pandas DataFrame with 12 columns
              - Can be a window/buffer of sensor readings (e.g., 3 seconds @ 50Hz = 150 rows)
              - Columns represent: [IMU0_acc_x, IMU0_acc_y, IMU0_acc_z, IMU0_gyro_x, ..., IMU1_gyro_z]
        col_order: Column names to use when converting numpy array to DataFrame
        timing: If True, print the preprocessing time
        
    Returns:
        Dictionary with 84 features (ready for model inference)
        
    Example for streaming:
        # Accumulate sensor readings in a buffer
        buffer = []  # List of [ax0, ay0, az0, gx0, gy0, gz0, ax1, ay1, az1, gx1, gy1, gz1]
        
        # When buffer is full (e.g., 3 seconds of data)
        data_array = np.array(buffer)  # Shape: (N, 12)
        features = features_from_12xN(data_array)  # Returns 84 features
        
        # Use features for model prediction
        # prediction = model.predict(features)
    """
    start_time = time.time()
    
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] != 12:
            raise ValueError(f"ndarray must be shape (N,12); got {data.shape}")
        df_window = pd.DataFrame(data, columns=list(col_order))
    elif isinstance(data, pd.DataFrame):
        missing = [c for c in DEFAULT_COL_ORDER if c not in data.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing[:5]}{'...' if len(missing)>5 else ''}")
        # Keep only the required columns (in canonical order)
        df_window = data.loc[:, DEFAULT_COL_ORDER]
    else:
        raise TypeError("`data` must be a pandas DataFrame or a NumPy ndarray")

    # Delegate to feature extraction
    result = extract_84_features(df_window)
    
    if timing:
        end_time = time.time()
        print(f"Preprocessing time: {end_time - start_time:.4f} seconds")
    
    return result

# =============================================================================
# CORE PREPROCESSING - END
# Everything above is needed for live streaming. Everything below is not needed.
# =============================================================================


# =============================================================================
# BATCH PROCESSING UTILITIES
# The functions below are for offline batch processing of CSV files.
# Not required for live streaming applications.
# =============================================================================

def process_directory(
    input_dir: str,
    output_csv: str,
    labels_json: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process all CSV files in a directory and extract 84 features from each.
    
    Args:
        input_dir: Path to directory containing CSV files with sensor data
        output_csv: Path where the output CSV with features will be saved
        labels_json: Optional path to JSON file mapping filenames to labels
        verbose: If True, print progress information
        
    Returns:
        DataFrame containing extracted features for all files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Load labels if provided
    labels_map = {}
    if labels_json:
        labels_path = Path(labels_json)
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                labels_map = json.load(f)
            if verbose:
                print(f"Loaded {len(labels_map)} labels from {labels_json}")
        else:
            print(f"Warning: Labels file not found: {labels_json}")
    
    # Find all CSV files
    csv_files = sorted(input_path.glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    results = []
    errors = []
    
    for i, csv_file in enumerate(csv_files, 1):
        if verbose and i % 50 == 0:
            print(f"Processing {i}/{len(csv_files)}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Extract features
            features = features_from_12xN(df)
            
            # Add metadata
            features['filename'] = csv_file.name
            
            # Extract file key (remove _data suffix if present)
            file_key = csv_file.stem.replace('_data', '')
            features['file_key'] = file_key
            
            # Add label if available
            if labels_map:
                features['label'] = labels_map.get(file_key, None)
            
            results.append(features)
            
        except Exception as e:
            errors.append((csv_file.name, str(e)))
            if verbose:
                print(f"Error processing {csv_file.name}: {e}")
    
    if verbose:
        print(f"\nSuccessfully processed {len(results)}/{len(csv_files)} files")
        if errors:
            print(f"Failed to process {len(errors)} files:")
            for fname, err in errors[:5]:  # Show first 5 errors
                print(f"  - {fname}: {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more")
    
    # Create DataFrame and save
    if not results:
        raise ValueError("No files were successfully processed")
    
    output_df = pd.DataFrame(results)
    
    # Reorder columns: metadata first, then features
    meta_cols = ['filename', 'file_key']
    if 'label' in output_df.columns:
        meta_cols.append('label')
    feature_cols = [c for c in output_df.columns if c not in meta_cols]
    output_df = output_df[meta_cols + feature_cols]
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\nSaved {len(output_df)} feature vectors to {output_csv}")
        print(f"Output shape: {output_df.shape} (rows × columns)")
    
    return output_df


def process_segmented_directory(
    input_dir: str,
    output_csv: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process segmented_data folder structure where each subdirectory represents a class.
    Label is extracted from the first character of the folder name (e.g., '1_left' -> label=1).
    
    Args:
        input_dir: Path to directory containing subdirectories with CSV files
        output_csv: Path where the output CSV with features will be saved
        verbose: If True, print progress information
        
    Returns:
        DataFrame containing filename, file_key, label, and 84 features for all files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {input_dir}")
    
    if verbose:
        print(f"Found {len(subdirs)} subdirectories to process")
    
    # Process each subdirectory
    results = []
    errors = []
    total_files = 0
    
    for subdir in sorted(subdirs):
        # Extract label from first character of folder name
        folder_name = subdir.name
        try:
            label = int(folder_name[0])
        except (ValueError, IndexError):
            if verbose:
                print(f"Warning: Could not extract label from folder '{folder_name}', skipping")
            continue
        
        # Find all CSV files in this subdirectory
        csv_files = sorted(subdir.glob('*.csv'))
        if not csv_files:
            if verbose:
                print(f"Warning: No CSV files found in {folder_name}")
            continue
        
        if verbose:
            print(f"\nProcessing {folder_name} (label={label}): {len(csv_files)} files")
        
        for i, csv_file in enumerate(csv_files, 1):
            total_files += 1
            if verbose and i % 50 == 0:
                print(f"  {i}/{len(csv_files)}...")
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Extract features
                features = features_from_12xN(df)
                
                # Add metadata
                features['filename'] = csv_file.name
                features['file_key'] = csv_file.stem
                features['label'] = label
                
                results.append(features)
                
            except Exception as e:
                errors.append((csv_file.name, str(e)))
                if verbose:
                    print(f"  Error processing {csv_file.name}: {e}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Successfully processed {len(results)}/{total_files} files")
        if errors:
            print(f"Failed to process {len(errors)} files:")
            for fname, err in errors[:5]:  # Show first 5 errors
                print(f"  - {fname}: {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more")
    
    # Create DataFrame and save
    if not results:
        raise ValueError("No files were successfully processed")
    
    output_df = pd.DataFrame(results)
    
    # Reorder columns: metadata first (filename, file_key, label), then features
    meta_cols = ['filename', 'file_key', 'label']
    feature_cols = [c for c in output_df.columns if c not in meta_cols]
    output_df = output_df[meta_cols + feature_cols]
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\nSaved {len(output_df)} feature vectors to {output_csv}")
        print(f"Output shape: {output_df.shape} (rows × columns)")
        print(f"Labels found: {sorted(output_df['label'].unique())}")
    
    return output_df


def scale_features_csv(
    input_csv: str,
    output_csv: str,
    scaler_path: Optional[str] = None,
    scaler: Optional[object] = None,
    meta_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    Scale features in a CSV file using a pre-fitted scaler.
    
    Args:
        input_csv: Path to CSV file with features
        output_csv: Path to save scaled features CSV
        scaler_path: Path to saved scaler (joblib/pickle file)
        scaler: Pre-fitted scaler object (if not loading from file)
        meta_cols: List of metadata columns to exclude from scaling (default: ['filename', 'file_key', 'label'])
        
    Returns:
        DataFrame with scaled features
    """
    if scaler is None:
        if scaler_path is None:
            raise ValueError("Either scaler_path or scaler must be provided")
        if joblib is None:
            raise ImportError("joblib required to load scaler. Install with: pip install joblib")
        scaler = joblib.load(scaler_path)
    
    df = pd.read_csv(input_csv)
    
    if meta_cols is None:
        meta_cols = ['filename', 'file_key', 'label']
    
    # Separate metadata and feature columns
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Scale features
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Create output DataFrame
    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled
    
    # Save
    df_scaled.to_csv(output_csv, index=False)
    return df_scaled


# =============================================================================
# COMMAND-LINE INTERFACE
# This section is only for running the script from command line.
# Not needed for importing and using in other Python code.
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract 84-dimensional features from IMU sensor data CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a directory with labels
  python preprocess.py --input test/data_clean/ --labels test/labels.json --output features.csv
  
  # Process without labels
  python preprocess.py --input test/data_clean/ --output features.csv
  
  # Run timing test
  python preprocess.py --test
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory containing CSV files with sensor data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file path for extracted features'
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        default=None,
        help='Optional JSON file mapping filenames to labels'
    )
    parser.add_argument(
        '--segmented',
        action='store_true',
        help='Process segmented data folder structure (labels from folder names)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run timing test instead of processing files'
    )
    parser.add_argument(
        '--scale',
        type=str,
        help='Scale features using saved scaler (provide scaler path)'
    )
    parser.add_argument(
        '--scale-output',
        type=str,
        help='Output path for scaled features (use with --scale)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run timing test
        print("Testing preprocessing timing...")
        arr = np.random.randn(150, 12)  # e.g., 3s@50Hz, 2 IMUs
        feats = features_from_12xN(arr, timing=True)
        print(f"Number of features: {len(feats)}")
    elif args.scale:
        # Scale features
        if not args.input:
            parser.error("--input required when using --scale")
        if not args.scale_output:
            args.scale_output = args.input.replace('.csv', '_scaled.csv')
        scale_features_csv(args.input, args.scale_output, scaler_path=args.scale)
        print(f"Scaled features saved to {args.scale_output}")
    elif args.input and args.output:
        if args.segmented:
            # Process segmented directory structure
            process_segmented_directory(
                input_dir=args.input,
                output_csv=args.output,
                verbose=not args.quiet
            )
        else:
            # Process flat directory
            process_directory(
                input_dir=args.input,
                output_csv=args.output,
                labels_json=args.labels,
                verbose=not args.quiet
            )
    else:
        parser.print_help()