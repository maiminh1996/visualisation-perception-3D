from mayavi.mlab import *
import numpy as np

K=10
xx = np.arange(0,K,1)
yy = np.arange(0,K,1)
x, y = np.meshgrid(xx, yy)
x, y = x.flatten(), y.flatten()
z = np.zeros(K*K)

colors = 1.0*(x+y)/(max(x)+max(y))
print(colors)
nodes = points3d(x,y,z, scale_factor=0.5)
nodes.glyph.scale_mode = 'scale_by_vector'

nodes.mlab_source.dataset.point_data.scalars = colors
show()
