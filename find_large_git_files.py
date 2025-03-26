import os
import subprocess

# Set file size threshold (100MB = 100 * 1024 * 1024 bytes)
SIZE_THRESHOLD = 100 * 1024 * 1024

def get_git_files():
    """Get a list of all files tracked by Git."""
    try:
        output = subprocess.check_output(["git", "ls-files"], text=True)
        return output.strip().split("\n")
    except subprocess.CalledProcessError:
        return []

def find_large_files():
    """Find and print files larger than SIZE_THRESHOLD."""
    large_files = []
    for file in get_git_files():
        if os.path.exists(file):
            size = os.path.getsize(file)
            if size > SIZE_THRESHOLD:
                large_files.append((file, size / (1024 * 1024)))  # Convert to MB

    if large_files:
        print("\n🚨 Large Files in Your Repository:")
        for file, size in sorted(large_files, key=lambda x: x[1], reverse=True):
            print(f"📁 {file} - {size:.2f} MB")
    else:
        print("✅ No large files found.")

if __name__ == "__main__":
    find_large_files()
