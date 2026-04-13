try:
    import sys
    print(f"Python version: {sys.version}")
except ImportError:
    print("✗ Python not found")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not found")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError:
    print("✗ Pandas not found")
try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ Scikit-learn not found")
try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except ImportError:
    print("✗ SciPy not found")
try:
    import networkx as nx
    print(f"✓ NetworkX {nx.__version__}")
except ImportError:
    print("✗ NetworkX not found")
try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib not found")
try:
    import open3d as o3d
    print(f"✓ Open3D {o3d.__version__}")
except ImportError:
    print("✗ Open3D not found")
try:
    from pxr import Usd
    print("✓ USD-Core available")
except ImportError:
    print("✗ USD-Core not found (optional)")
print("\nSetup verification complete!")
