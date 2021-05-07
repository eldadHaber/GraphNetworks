from e3nn.o3 import Irreps
import torch
from e3nn import o3

#We want to convert a vector to a spherical harmonics basis

v = torch.tensor([1,2,3.0])
# v = torch.tensor([1.0,0,0.0])

irreps = o3.Irreps("1x0e+1x1o+1x2e+1x3o+1x4e")
v_sh = o3.spherical_harmonics(irreps, v, False, normalization='component')





irreps = Irreps("1o")
print(irreps)
print('done')

# Tuple[Tuple[int, Tuple[int, int]]]
# ((multiplicity, (l, p)), ...)

print(len(irreps))
mul_ir = irreps[0]  # a tuple

print(mul_ir)
print(len(mul_ir))
mul = mul_ir[0]  # an int
ir = mul_ir[1]  # another tuple

print(mul)

print(ir)
# print(len(ir))  ir is a tuple of 2 ints but __len__ has been disabled since it is always 2
l = ir[0]
p = ir[1]

print(l, p)


import torch
t = torch.tensor

# show the transformation matrix corresponding to the inversion
inv = irreps.D_from_angles(alpha=t(0.0), beta=t(0.0), gamma=t(0.0), k=t(1))

# a small rotation around the y axis
rot = irreps.D_from_angles(alpha=t(0.1), beta=t(0.0), gamma=t(0.0), k=t(0))


irreps = Irreps("7x0e + 3x0o + 5x1o + 5x2o")



from e3nn import o3
rot = -o3.rand_matrix()

D = irreps.D_from_matrix(rot)

import matplotlib.pyplot as plt
plt.imshow(D, cmap='bwr', vmin=-1, vmax=1);

