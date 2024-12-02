# PDistMap

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14257900.svg)](https://doi.org/10.5281/zenodo.14257900)

This package calculates the overlap percentage between two probability distributions, offering extensive applications in both academic and industrial settings. For instance, in multiple iterations of machine learning clustering, the core algorithm may change the cluster number or name, making it challenging for the end user to map the clusters accurately.

### Example Use Cases:

- **Machine Learning Clustering:** In scenarios where multiple iterations of clustering algorithms are performed, the cluster identifiers may change, making it difficult to track and compare clusters across iterations. This package helps in mapping and comparing clusters by calculating the overlap percentage between the distributions of cluster assignments. For example, if a data scientist is running a k-means clustering algorithm multiple times, the cluster labels might change in each iteration. By using this package, they can measure the overlap between the clusters from different iterations and ensure consistency in their analysis.

- **Anomaly Detection:** The package can be used to compare the distribution of data points in normal and anomalous conditions, helping in identifying and quantifying the extent of anomalies. For instance, in a network security application, the distribution of network traffic under normal conditions can be compared with the distribution during a suspected attack. The overlap percentage can help quantify the deviation and identify potential security breaches.

- **Quality Control:** In manufacturing and quality control processes, the package can be used to compare the distribution of measurements from different batches or production runs, ensuring consistency and identifying deviations. For example, a quality control engineer can compare the distribution of product dimensions from two different production runs to ensure that they meet the required specifications and identify any deviations that need to be addressed.

- **Market Research:** The package can be applied to compare the distribution of survey responses or customer preferences across different demographic groups or time periods, providing insights into market trends and changes in consumer behavior. For instance, a market researcher can compare the distribution of customer satisfaction scores from two different regions to identify any significant differences and tailor marketing strategies accordingly.

- **Healthcare Analytics:** In healthcare, the package can be used to compare the distribution of patient outcomes or treatment responses across different groups, aiding in the evaluation of treatment effectiveness and identifying potential disparities. For example, a healthcare analyst can compare the distribution of recovery times for patients receiving two different treatments to determine which treatment is more effective and identify any disparities in treatment outcomes.

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

