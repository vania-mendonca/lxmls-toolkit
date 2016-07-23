import numpy as np

a = np.random.rand(10)

print np.log(sum(np.exp(a)))
print np.log(sum(np.exp(10 * a)))
print np.log(sum(np.exp(100 * a)))
print np.log(sum(np.exp(1000 * a)))

from lxmls.sequences.log_domain import *

print logsum(a)
print logsum(10 * a)
print logsum(100 * a)
print logsum(1000 * a)

# Output
# 2.81571949409
# 10.5567261544
# 96.2372462856
# inf
# 2.81571949409
# 10.5567261544
# 96.2372462856
# 958.806858837
# ./day2/Ex2_4.py:8: RuntimeWarning: overflow encountered in exp
#   np.log(sum(np.exp(1000 * a)))
