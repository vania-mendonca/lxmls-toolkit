# import sys
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append('./lxmls/readers')

import lxmls.readers.galton as galton

galton_data = galton.load()

print np.mean(galton_data)
print np.std(galton_data)

# plt.hist(galton_data)
# plt.show()

d = np.ravel(galton_data)
# plt.hist(d)
# plt.show()

dim1, dim2 = galton_data.shape

r = galton_data + np.random.randn(dim1, dim2)
plt.hist(r)
plt.show()
