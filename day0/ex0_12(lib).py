import numpy as np

import lxmls.readers.galton as galton

galton_data = galton.load()

X = np.empty_like(galton_data)
X[:, 0] = np.ones_like(galton_data[:, 0])
X[:, 1] = galton_data[:, 0]

ret = np.linalg.lstsq(X, galton_data[:, 1])

print ret