#!/usr/bin/env python3
"""
IMU Gesture Segmentation Script

Analyzes IMU sensor data to automatically detect gesture patterns using variance-based
detection and segments them into 1-second windows for model training.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class IMUGestureSegmenter:
    """Segments IMU data into gesture windows using variance-based detection."""
    
    def __init__(
        self,
        window_size: float = 1.0,
        variance_window: float = 0.2,
        threshold: Optional[float] = None,
        min_gap: float = 0.3
    ):
        """
        Initialize the segmenter.
        
        Args:
            window_size: Duration of each output window in seconds
            variance_window: Window size for rolling variance calculation in seconds
            threshold: Variance threshold for activity detection (None for auto)
            min_gap: Minimum gap between gestures in seconds
        """
        self.window_size = window_size
        self.variance_window = variance_window
        self.threshold = threshold
        self.min_gap = min_gap
        
    def load_csv(self, filepath: str) -> Tuple[pd.DataFrame, str]:
        """
        Load IMU data from CSV file and extract gesture label from path.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple of (DataFrame with IMU data, gesture label)
        """
        # Load the CSV
        df = pd.read_csv(filepath)
        
        # Extract gesture label from folder structure
        path_parts = Path(filepath).parts
        gesture_label = "unknown"
        
        # Look for gesture folder name (e.g., "grasp" in /onx/grasp/)
        for i, part in enumerate(path_parts):
            if part == "onx" and i + 1 < len(path_parts):
                gesture_label = path_parts[i + 1]
                break
        
        print(f"Loaded {len(df)} samples from {filepath}")
        print(f"Detected gesture label: {gesture_label}")
        
        return df, gesture_label
    
    def calculate_sampling_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate the sampling rate from timestamps.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            Sampling rate in Hz
        """
        if len(df) < 2:
            raise ValueError("Need at least 2 samples to calculate sampling rate")
        
        # Calculate time differences (timestamps are in milliseconds)
        time_diffs = df['timestamp'].diff().dropna()
        avg_diff_ms = time_diffs.mean()
        
        # Convert to Hz
        sampling_rate = 1000.0 / avg_diff_ms
        
        print(f"Detected sampling rate: {sampling_rate:.2f} Hz")
        print(f"Average sample interval: {avg_diff_ms:.2f} ms")
        
        return sampling_rate
    
    def calculate_activity_score(self, df: pd.DataFrame, sampling_rate: float) -> np.ndarray:
        """
        Calculate activity score using rolling variance across all IMU channels.
        
        Args:
            df: DataFrame with IMU data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Array of activity scores
        """
        # Get all IMU columns (exclude timestamp)
        imu_columns = [col for col in df.columns if col != 'timestamp']
        
        # Calculate window size in samples
        window_samples = max(int(self.variance_window * sampling_rate), 3)
        
        print(f"Calculating variance with window of {window_samples} samples (~{self.variance_window}s)")
        
        # Calculate rolling variance for each channel
        variances = []
        for col in imu_columns:
            var = df[col].rolling(window=window_samples, center=True).var()
            variances.append(var.fillna(0))
        
        # Aggregate activity score (mean of all channel variances)
        activity_score = np.mean(variances, axis=0)
        
        return activity_score
    
    def detect_gesture_regions(
        self,
        activity_score: np.ndarray,
        sampling_rate: float,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int]]:
        """
        Detect gesture regions based on activity threshold.
        
        Args:
            activity_score: Array of activity scores
            sampling_rate: Sampling rate in Hz
            threshold: Threshold value (None for auto)
            
        Returns:
            List of (start_idx, end_idx) tuples for gesture regions
        """
        # Auto-calculate threshold if not provided
        if threshold is None:
            # Use 3 * (median + 2 * MAD) for higher threshold
            median = np.median(activity_score)
            mad = np.median(np.abs(activity_score - median))
            threshold = 3 * (median + 2 * mad)
            print(f"Auto-calculated threshold (3x): {threshold:.2f}")
        else:
            print(f"Using provided threshold: {threshold:.2f}")
        
        self.threshold = threshold
        
        # Identify samples above threshold
        above_threshold = activity_score > threshold
        
        # Find contiguous regions
        regions = []
        in_region = False
        start_idx = 0
        
        min_gap_samples = int(self.min_gap * sampling_rate)
        
        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_region:
                # Start of new region
                start_idx = i
                in_region = True
            elif not above_threshold[i] and in_region:
                # Check if this is a small gap or end of region
                # Look ahead to see if activity resumes within min_gap
                gap_end = min(i + min_gap_samples, len(above_threshold))
                if np.any(above_threshold[i:gap_end]):
                    # Activity resumes, don't end region yet
                    continue
                else:
                    # End of region
                    regions.append((start_idx, i))
                    in_region = False
        
        # Handle case where last region extends to end
        if in_region:
            regions.append((start_idx, len(above_threshold)))
        
        print(f"Detected {len(regions)} gesture regions")
        
        return regions
    
    def segment_into_windows(
        self,
        df: pd.DataFrame,
        regions: List[Tuple[int, int]],
        sampling_rate: float,
        activity_score: np.ndarray
    ) -> List[pd.DataFrame]:
        """
        Segment detected gesture regions into fixed-size windows centered around peaks.
        Ensures no overlaps between windows.
        
        Args:
            df: Full DataFrame with IMU data
            regions: List of (start_idx, end_idx) gesture regions
            sampling_rate: Sampling rate in Hz
            activity_score: Array of activity scores to find peaks
            
        Returns:
            List of DataFrames, each containing one window
        """
        window_samples = int(self.window_size * sampling_rate)
        windows = []
        used_ranges = []  # Track (start, end) of already extracted windows
        
        def has_overlap(start, end, used_ranges):
            """Check if a range overlaps with any used ranges."""
            for used_start, used_end in used_ranges:
                # Check for any overlap
                if not (end <= used_start or start >= used_end):
                    return True
            return False
        
        for start_idx, end_idx in regions:
            region_length = end_idx - start_idx
            
            # Find the peak (maximum activity) within this region
            region_activity = activity_score[start_idx:end_idx]
            peak_offset = np.argmax(region_activity)
            peak_idx = start_idx + peak_offset
            
            # If region is short, create one window centered on the peak
            if region_length <= window_samples * 1.5:
                # Center window around peak
                half_window = window_samples // 2
                window_start = peak_idx - half_window
                window_end = window_start + window_samples
                
                # Adjust if window goes out of bounds
                if window_start < 0:
                    window_start = 0
                    window_end = window_samples
                elif window_end > len(df):
                    window_end = len(df)
                    window_start = max(0, window_end - window_samples)
                
                # Check for overlap
                if not has_overlap(window_start, window_end, used_ranges):
                    window = df.iloc[window_start:window_end].copy()
                    if len(window) >= window_samples * 0.8:  # At least 80% of desired length
                        windows.append(window)
                        used_ranges.append((window_start, window_end))
            
            # If region is large, extract multiple windows centered on local peaks
            else:
                # Find multiple local peaks within the region
                # Use minimum distance equal to window size to prevent overlaps
                min_distance = window_samples
                
                # Simple peak detection: find local maxima
                peaks = []
                for i in range(start_idx + min_distance, end_idx, min_distance):
                    # Look at a small window around this point
                    search_start = max(start_idx, i - min_distance // 2)
                    search_end = min(end_idx, i + min_distance // 2)
                    local_region = activity_score[search_start:search_end]
                    
                    if len(local_region) > 0:
                        local_peak_offset = np.argmax(local_region)
                        local_peak_idx = search_start + local_peak_offset
                        peaks.append(local_peak_idx)
                
                # If no peaks found, just use the main peak
                if not peaks:
                    peaks = [peak_idx]
                
                # Create windows centered on each peak (no overlaps)
                for peak in peaks:
                    half_window = window_samples // 2
                    window_start = peak - half_window
                    window_end = window_start + window_samples
                    
                    # Adjust if window goes out of bounds
                    if window_start < 0:
                        window_start = 0
                        window_end = window_samples
                    elif window_end > len(df):
                        window_end = len(df)
                        window_start = max(0, window_end - window_samples)
                    
                    # Check for overlap before adding
                    margin = window_samples // 4
                    if (window_start >= start_idx - margin and 
                        window_end <= end_idx + margin and
                        not has_overlap(window_start, window_end, used_ranges)):
                        window = df.iloc[window_start:window_end].copy()
                        if len(window) >= window_samples * 0.8:
                            windows.append(window)
                            used_ranges.append((window_start, window_end))
        
        print(f"Extracted {len(windows)} non-overlapping windows of ~{self.window_size}s each")
        print(f"Each window is centered around a variance spike/peak")
        
        return windows
    
    def save_windows(
        self,
        windows: List[pd.DataFrame],
        gesture_label: str,
        output_dir: str
    ) -> List[str]:
        """
        Save segmented windows to individual CSV files.
        
        Args:
            windows: List of window DataFrames
            gesture_label: Gesture label for filename
            output_dir: Output directory path
            
        Returns:
            List of saved file paths
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the subfolder name for filename prefix
        subfolder_name = output_path.name
        
        saved_files = []
        
        for i, window in enumerate(windows, start=1):
            filename = f"{subfolder_name}_{gesture_label}_{i:03d}.csv"
            filepath = output_path / filename
            
            # Save with header, preserving original timestamps
            window.to_csv(filepath, index=False)
            saved_files.append(str(filepath))
        
        print(f"\nSaved {len(saved_files)} files to {output_dir}/")
        
        return saved_files
    
    def visualize(
        self,
        df: pd.DataFrame,
        activity_score: np.ndarray,
        regions: List[Tuple[int, int]],
        sampling_rate: float,
        windows: List[pd.DataFrame] = None
    ):
        """
        Visualize the activity score and detected gesture regions.
        
        Args:
            df: Full DataFrame with IMU data
            activity_score: Array of activity scores
            regions: List of detected gesture regions
            sampling_rate: Sampling rate in Hz
            windows: Optional list of extracted windows to visualize
        """
        # Create time axis in seconds
        time_s = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Activity score with threshold and detected regions
        axes[0].plot(time_s, activity_score, label='Activity Score', linewidth=1)
        axes[0].axhline(y=self.threshold, color='r', linestyle='--', 
                       label=f'Threshold ({self.threshold:.2f})')
        
        # Highlight detected regions
        for start_idx, end_idx in regions:
            axes[0].axvspan(time_s.iloc[start_idx], time_s.iloc[end_idx-1], 
                           alpha=0.3, color='green', label='Detected Gesture' if start_idx == regions[0][0] else '')
        
        # Mark peaks within regions
        for start_idx, end_idx in regions:
            region_activity = activity_score[start_idx:end_idx]
            peak_offset = np.argmax(region_activity)
            peak_idx = start_idx + peak_offset
            axes[0].plot(time_s.iloc[peak_idx], activity_score[peak_idx], 
                        'r*', markersize=15, 
                        label='Peak Center' if start_idx == regions[0][0] else '',
                        zorder=5)
        
        # Show extracted window boundaries if provided
        if windows is not None and len(windows) > 0:
            for i, window in enumerate(windows):
                window_start_time = (window['timestamp'].iloc[0] - df['timestamp'].iloc[0]) / 1000.0
                window_end_time = (window['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000.0
                
                # Draw vertical lines for window boundaries
                axes[0].axvline(x=window_start_time, color='blue', linestyle=':', 
                              linewidth=1.5, alpha=0.6,
                              label='Window Boundaries' if i == 0 else '')
                axes[0].axvline(x=window_end_time, color='blue', linestyle=':', 
                              linewidth=1.5, alpha=0.6)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Activity Score (Variance)')
        axes[0].set_title('Activity Detection and Gesture Regions (with Peak Centers)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Sample IMU channels
        imu_cols = [col for col in df.columns if col != 'timestamp'][:6]  # First 6 channels
        for col in imu_cols:
            axes[1].plot(time_s, df[col], label=col, linewidth=0.8, alpha=0.7)
        
        # Highlight detected regions
        for start_idx, end_idx in regions:
            axes[1].axvspan(time_s.iloc[start_idx], time_s.iloc[end_idx-1], 
                           alpha=0.2, color='green')
        
        # Show extracted window boundaries if provided
        if windows is not None and len(windows) > 0:
            for i, window in enumerate(windows):
                window_start_time = (window['timestamp'].iloc[0] - df['timestamp'].iloc[0]) / 1000.0
                window_end_time = (window['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000.0
                
                # Draw vertical lines for window boundaries
                axes[1].axvline(x=window_start_time, color='blue', linestyle=':', 
                              linewidth=1.5, alpha=0.6)
                axes[1].axvline(x=window_end_time, color='blue', linestyle=':', 
                              linewidth=1.5, alpha=0.6)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Sensor Value')
        axes[1].set_title('Sample IMU Channels (IMU0) with 1-Second Window Boundaries')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def process(
        self,
        input_file: str,
        output_dir: str,
        visualize: bool = False
    ) -> List[str]:
        """
        Complete processing pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Output directory for segmented files
            visualize: Whether to show visualization
            
        Returns:
            List of saved file paths
        """
        print(f"\n{'='*60}")
        print("IMU Gesture Segmentation")
        print(f"{'='*60}\n")
        
        # Load data
        df, gesture_label = self.load_csv(input_file)
        
        # Calculate sampling rate
        sampling_rate = self.calculate_sampling_rate(df)
        
        # Calculate activity score
        activity_score = self.calculate_activity_score(df, sampling_rate)
        
        # Detect gesture regions
        regions = self.detect_gesture_regions(activity_score, sampling_rate, self.threshold)
        
        if len(regions) == 0:
            print("\nWARNING: No gesture regions detected!")
            print("Try adjusting the threshold or check your data.")
            return []
        
        # Segment into windows (centered around peaks)
        windows = self.segment_into_windows(df, regions, sampling_rate, activity_score)
        
        if len(windows) == 0:
            print("\nWARNING: No valid windows extracted!")
            return []
        
        # Display window statistics
        print(f"\nWindow Statistics:")
        window_lengths = [len(w) for w in windows]
        print(f"  Min samples: {min(window_lengths)}")
        print(f"  Max samples: {max(window_lengths)}")
        print(f"  Mean samples: {np.mean(window_lengths):.1f}")
        print(f"  Target samples: {int(self.window_size * sampling_rate)}")
        
        # Save windows
        saved_files = self.save_windows(windows, gesture_label, output_dir)
        
        # Visualize if requested
        if visualize:
            self.visualize(df, activity_score, regions, sampling_rate, windows)
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}\n")
        
        return saved_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Segment IMU gesture data into training windows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto threshold
  python segment_gestures.py input.csv --output-dir ./segmented/
  
  # With visualization
  python segment_gestures.py input.csv -o ./out/ --visualize
  
  # Custom threshold and window size
  python segment_gestures.py input.csv -o ./out/ --threshold 5000 --window-size 1.5
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file with IMU data'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./segmented_gestures',
        help='Output directory for segmented files (default: ./segmented_gestures)'
    )
    
    parser.add_argument(
        '--threshold',
        type=str,
        default='auto',
        help='Variance threshold for gesture detection (use "auto" for automatic, default: auto)'
    )
    
    parser.add_argument(
        '--window-size',
        type=float,
        default=1.0,
        help='Duration of each output window in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--variance-window',
        type=float,
        default=0.2,
        help='Window size for variance calculation in seconds (default: 0.2)'
    )
    
    parser.add_argument(
        '--min-gap',
        type=float,
        default=0.3,
        help='Minimum gap between gestures in seconds (default: 0.3)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of detected gestures'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Parse threshold
    threshold = None if args.threshold.lower() == 'auto' else float(args.threshold)
    
    # Create segmenter
    segmenter = IMUGestureSegmenter(
        window_size=args.window_size,
        variance_window=args.variance_window,
        threshold=threshold,
        min_gap=args.min_gap
    )
    
    # Process the file
    saved_files = segmenter.process(
        args.input_file,
        args.output_dir,
        visualize=args.visualize
    )
    
    if saved_files:
        print(f"Successfully segmented {len(saved_files)} gesture windows!")
    else:
        print("No gestures were extracted. Try adjusting parameters.")
        sys.exit(1)


if __name__ == '__main__':
    main()

