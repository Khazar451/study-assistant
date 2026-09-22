import sys
from pathlib import Path

# Ensure the project root is in sys.path so 'src' can be imported anywhere
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
