#!/usr/bin/env python3
"""
Sample subdirectories from renders folder with interval of 4.
Extract every 4th subdirectory and save absolute paths to a txt file.
"""

import os
from pathlib import Path

# Source directory
source_dir = Path('/disk2/licheng/code/ARIN5201-CV-FinalProject/datasets/Toys2h/renders')

# Verify directory exists
if not source_dir.exists():
    print(f'Error: Directory does not exist: {source_dir}')
    exit(1)

if not source_dir.is_dir():
    print(f'Error: Path is not a directory: {source_dir}')
    exit(1)

# Get all subdirectories and sort them
subdirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
print(f'Total subdirectories: {len(subdirs)}')

# Sample every 4th subdirectory (interval = 4)
sampled_dirs = subdirs[::4]
print(f'Sampled subdirectories (interval=4): {len(sampled_dirs)}')

# Get absolute paths
absolute_paths = [str(d.resolve()) for d in sampled_dirs]

# Output file path
output_file = Path('/disk2/licheng/code/ARIN5201-CV-FinalProject/extract_test_paths.txt')

# Write to file
with open(output_file, 'w') as f:
    for path in absolute_paths:
        f.write(path + '\n')

print(f'\nResults saved to: {output_file}')
print(f'Total paths written: {len(absolute_paths)}')
print('\nFirst 5 paths:')
for path in absolute_paths[:5]:
    print(f'  {path}')
if len(absolute_paths) > 5:
    print(f'  ...')
    print(f'Last path:')
    print(f'  {absolute_paths[-1]}')
