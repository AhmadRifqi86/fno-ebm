#!/usr/bin/env python3
"""
Memory-Efficient HDF5 File Merger
Combines multiple large HDF5 files without overwhelming RAM using chunked processing
"""

import h5py
import numpy as np
import sys
import os
from pathlib import Path

def get_file_size_gb(filename):
    """Get file size in GB"""
    return os.path.getsize(filename) / (1024**3)

def merge_hdf5_files(input_files, output_file, chunk_size=1000):
    """
    Merge multiple HDF5 files into one using memory-efficient chunked copying.
    
    Parameters:
    -----------
    input_files : list
        List of input HDF5 file paths
    output_file : str
        Path to output merged HDF5 file
    chunk_size : int
        Number of rows to process at a time (adjust based on your RAM)
    """
    
    print(f"\n{'='*80}")
    print("HDF5 File Merger - Memory Efficient Mode")
    print(f"{'='*80}\n")
    
    # Validate input files
    for i, fname in enumerate(input_files, 1):
        if not os.path.exists(fname):
            print(f"❌ Error: File {fname} does not exist!")
            sys.exit(1)
        size_gb = get_file_size_gb(fname)
        print(f"✓ File {i}: {fname} ({size_gb:.2f} GB)")
    
    print(f"\nOutput file: {output_file}")
    print(f"Chunk size: {chunk_size} rows")
    print()
    
    # Open first file to inspect structure
    print("Inspecting structure of first file...")
    with h5py.File(input_files[0], 'r') as f:
        datasets = {}
        groups = []
        
        def collect_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'chunks': obj.chunks,
                    'compression': obj.compression
                }
            elif isinstance(obj, h5py.Group):
                groups.append(name)
        
        f.visititems(collect_items)
        
        print(f"Found {len(datasets)} dataset(s) and {len(groups)} group(s)\n")
        
        for name, info in datasets.items():
            print(f"Dataset: {name}")
            print(f"  Shape: {info['shape']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Chunks: {info['chunks']}")
            print(f"  Compression: {info['compression']}")
            print()
    
    # Create output file and merge
    print(f"{'='*80}")
    print("Starting merge process...")
    print(f"{'='*80}\n")
    
    with h5py.File(output_file, 'w') as out_f:
        
        # Create groups first
        for group_name in groups:
            out_f.create_group(group_name)
            print(f"✓ Created group: {group_name}")
        
        # Process each dataset
        for ds_name in datasets.keys():
            print(f"\n📊 Processing dataset: {ds_name}")
            print("-" * 80)
            
            # Determine concatenation axis (usually axis 0)
            concat_axis = 0
            
            # Get total size across all files
            total_size = 0
            shapes = []
            
            for fname in input_files:
                with h5py.File(fname, 'r') as f:
                    ds = f[ds_name]
                    shapes.append(ds.shape)
                    total_size += ds.shape[concat_axis]
            
            print(f"Individual file shapes: {shapes}")
            
            # Create output dataset with full size
            first_shape = shapes[0]
            output_shape = list(first_shape)
            output_shape[concat_axis] = total_size
            output_shape = tuple(output_shape)
            
            print(f"Output shape: {output_shape}")
            
            # Get dataset properties from first file
            with h5py.File(input_files[0], 'r') as f:
                ds = f[ds_name]
                dtype = ds.dtype
                chunks = ds.chunks
                compression = ds.compression
            
            # Create output dataset with same properties
            out_ds = out_f.create_dataset(
                ds_name,
                shape=output_shape,
                dtype=dtype,
                chunks=chunks,
                compression=compression if compression else None
            )
            
            print(f"Created output dataset")
            
            # Copy data from each file chunk by chunk
            write_offset = 0
            
            for file_idx, fname in enumerate(input_files, 1):
                print(f"\n  Copying from file {file_idx}/{len(input_files)}: {Path(fname).name}")
                
                with h5py.File(fname, 'r') as in_f:
                    in_ds = in_f[ds_name]
                    num_items = in_ds.shape[concat_axis]
                    
                    # Process in chunks to avoid loading entire dataset into RAM
                    num_chunks = (num_items + chunk_size - 1) // chunk_size
                    
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, num_items)
                        
                        # Read chunk
                        if len(in_ds.shape) == 1:
                            chunk_data = in_ds[start_idx:end_idx]
                        elif len(in_ds.shape) == 2:
                            chunk_data = in_ds[start_idx:end_idx, :]
                        else:
                            # Handle higher dimensional datasets
                            slices = [slice(None)] * len(in_ds.shape)
                            slices[concat_axis] = slice(start_idx, end_idx)
                            chunk_data = in_ds[tuple(slices)]
                        
                        # Write chunk to output
                        write_start = write_offset + start_idx
                        write_end = write_offset + end_idx
                        
                        if len(out_ds.shape) == 1:
                            out_ds[write_start:write_end] = chunk_data
                        elif len(out_ds.shape) == 2:
                            out_ds[write_start:write_end, :] = chunk_data
                        else:
                            slices = [slice(None)] * len(out_ds.shape)
                            slices[concat_axis] = slice(write_start, write_end)
                            out_ds[tuple(slices)] = chunk_data
                        
                        # Progress indicator
                        progress = (chunk_idx + 1) / num_chunks * 100
                        print(f"    Progress: {progress:.1f}% ({end_idx}/{num_items} items)", end='\r')
                    
                    print()  # New line after progress
                    write_offset += num_items
            
            print(f"✓ Completed dataset: {ds_name}")
        
        # Copy attributes from first file
        print(f"\n{'='*80}")
        print("Copying attributes...")
        with h5py.File(input_files[0], 'r') as first_f:
            # Copy root attributes
            for attr_name, attr_value in first_f.attrs.items():
                out_f.attrs[attr_name] = attr_value
                print(f"✓ Copied root attribute: {attr_name}")
            
            # Copy dataset attributes
            for ds_name in datasets.keys():
                for attr_name, attr_value in first_f[ds_name].attrs.items():
                    out_f[ds_name].attrs[attr_name] = attr_value
                    print(f"✓ Copied attribute: {ds_name}.{attr_name}")
    
    # Final summary
    output_size_gb = get_file_size_gb(output_file)
    print(f"\n{'='*80}")
    print("✅ MERGE COMPLETE!")
    print(f"{'='*80}")
    print(f"Output file: {output_file}")
    print(f"Output size: {output_size_gb:.2f} GB")
    print(f"Total input size: {sum(get_file_size_gb(f) for f in input_files):.2f} GB")
    print()

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python merge_hdf5.py output.hdf5 input1.hdf5 input2.hdf5 input3.hdf5 input4.hdf5 [chunk_size]")
        print("\nOptions:")
        print("  chunk_size: Number of rows to process at once (default: 1000)")
        print("              Decrease if running out of RAM, increase for faster processing")
        sys.exit(1)
    
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    
    # Check if last argument is chunk_size
    chunk_size = 1000
    if input_files[-1].isdigit():
        chunk_size = int(input_files[-1])
        input_files = input_files[:-1]
    
    # Validate we have input files
    if len(input_files) < 2:
        print("Error: Need at least 2 input files to merge")
        sys.exit(1)
    
    merge_hdf5_files(input_files, output_file, chunk_size=chunk_size)