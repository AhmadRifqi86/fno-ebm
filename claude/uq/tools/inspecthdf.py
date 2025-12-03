#!/usr/bin/env python3
"""
HDF5 File Structure Inspector
Inspects the structure of an HDF5 file to understand its contents before merging
"""

import h5py
import sys

def inspect_hdf5(filename):
    """Recursively inspect HDF5 file structure"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {filename}")
    print(f"{'='*80}\n")
    
    with h5py.File(filename, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: {name}")
                print(f"{indent}   Shape: {obj.shape}")
                print(f"{indent}   Dtype: {obj.dtype}")
                print(f"{indent}   Size: {obj.size * obj.dtype.itemsize / (1024**3):.2f} GB")
                if obj.chunks:
                    print(f"{indent}   Chunks: {obj.chunks}")
                print()
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: {name}")
        
        print("File structure:")
        print("-" * 80)
        f.visititems(print_structure)
        
        print("\nRoot level keys:")
        print("-" * 80)
        for key in f.keys():
            print(f"  • {key}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_hdf5.py <file.hdf5>")
        sys.exit(1)
    
    inspect_hdf5(sys.argv[1])