"""
NEURON Test Configuration.

Sets up the Python path so tests can import neuron modules
without requiring `pip install -e .` or sys.path hacks.
"""

import sys
from pathlib import Path

# Add src/ to path so `import neuron` works in tests
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
