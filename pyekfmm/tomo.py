def formL2d(paths, ax=[0,0.01,101],ay=[0,0.01,101]):
	"""
	formL2d: form L matrix for tomography
	
	INPUT
	%paths: [2 x ngrid]
	
	OUTPUT
	L: 	forward matrix
	"""
	ngrid=paths.shape[1]
# 	start_points=np.expand_dims(start_points,1);
	
	
	nx=ax[2];ny=ay[2];
	dx=ax[1];dy=ay[1];
	ox=ax[0];oy=ay[0];
	
	L=np.zeros([1,nx*ny])
	for ix in range(0,nx):
		for iy in range(0,ny):
			x0=ox+ix*dx
			x1=ox+(ix+1)*dx
			y0=oy+iy*dy
			y1=oy+(iy+1)*dy
			
			
			
			L[ix+iy*nx]=

		

	
	return L