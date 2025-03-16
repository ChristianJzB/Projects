import sys
import os

# Get the absolute path to the project root (adjust as needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

print(project_root)

# Add the project root and Elliptic directory to sys.path
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "elliptic"))  # Explicitly add Elliptic folder

# Print sys.path to debug
print("Updated sys.path:", sys.path)

# Now import elliptic_files
import elliptic_files