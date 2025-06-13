import os
from pathlib import Path

def get_readable_size(size_bytes):
    """Convert bytes to a human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / 1024**2:.2f} MB"

def print_dill_file_sizes(directory):
    """Print sizes of all .dill files in the given directory (recursively)."""
    directory = Path(directory)
    dill_files = list(directory.rglob("*.dill"))  # Recursive search

    if not dill_files:
        print("No .dill files found.")
        return

    print(f"\nFound {len(dill_files)} .dill files in '{directory}':\n")
    for path in dill_files:
        size = path.stat().st_size
        print(f"{path}: {get_readable_size(size)}")

# Example usage:
if __name__ == "__main__":
    folder_path = "jsindy_results"  # Change to your target directory
    print_dill_file_sizes(folder_path)
