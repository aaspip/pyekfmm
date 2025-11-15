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
	
def vgrad3d(v1,v2,n1,n2,n3):
	'''
	vgrad: velocity model with gradient increase
	
	INPUT
	v1,v2: starting and ending velocities
	n1,n2,n3: axis size (z,x,y)
	
	OUTPUT
	vel
	
	EXAMPLE
	from pyekfmm import vgrad3d
	import matplotlib.pyplot as plt
	vel=vgrad3d(3,5,50,40,30)
	from pyseistr import plot3d
	plot3d(vel,dz=1,dx=1,dy=1,cmap=plt.cm.jet,barlabel='Velocity (km/s)',showf=False,close=False)
	plt.title('Velocity model of linear increase with depth')
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.show()
	'''
	
	import numpy as np
	
	vel=np.matmul(np.linspace(v1,v2,n1)[:,np.newaxis],np.ones([1,n2]))
	
	vel3d=np.zeros([n1,n2,n3])
	for ii in range(n3):
		vel3d[:,:,ii]=vel
	
	return vel3d