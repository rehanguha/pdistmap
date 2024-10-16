
# PDistMap

This package helps to find the overlap percentage of two probability distributions.

## Installation

```bash
pip install pdistmap
```

## How to use it

### Method 1

```python

from pdistmap.set import KDEIntersection
import numpy as np

A = np.array([25, 40, 70, 65, 69, 75, 80, 85])
B = np.array([25, 40, 70, 65, 69, 75, 80, 85, 81, 90])

area = KDEIntersection(A,B).intersection_area()
print(area) # Expected output: 0.8752770150023454


KDEIntersection(A,B).intersection_area(plot = True)

```

![Sample Image](artifact/KDE_Plot.png)

