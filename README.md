# PDistMap

This package helps to find the overlap percentage of two probability distributions.

## Installation

```bash
pip install pdistmap
```

## How to use it

```python

from pdistmap.intersection import KDEIntersection
import numpy as np

A = np.array([25, 40, 70, 65, 69, 75, 80, 85])
B = np.array([25, 40, 70, 65, 69, 75, 80, 85, 81, 90])

area = KDEIntersection(A,B).intersection_area()
print(area) # Expected output: 0.8752770150023454

```