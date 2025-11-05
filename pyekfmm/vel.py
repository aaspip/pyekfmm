def vgrad(v1,v2,n1,n2):
	'''
	vgrad: velocity model with gradient increase
	
	INPUT
	v1,v2: starting and ending velocities
	n1,n2: axis size
	
	OUTPUT
	vel
	
	EXAMPLE
	from pyekfmm import vgrad
	import matplotlib.pyplot as plt
	
	vel=vgrad(3,5,50,40)
	plt.imshow(vel);
	plt.title('Velocity model of linear increase with depth')
	plt.xlabel('X sample');plt.ylabel('Z sample');
	plt.colorbar(label='Velocity (km/s)');
	plt.show()
	
	'''
	
	import numpy as np
	
	vel=np.matmul(np.linspace(v1,v2,n1)[:,np.newaxis],np.ones([1,n2]))
	
	return vel