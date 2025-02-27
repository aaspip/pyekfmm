import numpy as np

def formL2d(paths, ax=[0,0.01,101],ay=[0,0.01,101]):
	"""
	formL2d: form L matrix for tomography
	
	INPUT
	%paths: [2 x ngrid]
	%paths is a list of raypath arrays, npath=len(paths);
	
	OUTPUT
	L: 	forward matrix
	"""

	npath=len(paths);
	
	nx=ax[2];ny=ay[2];
	dx=ax[1];dy=ay[1];
	ox=ax[0];oy=ay[0];

	nlon=nx;nlat=ny;lonstep=dx;latstep=dy;lonmin=ox;latmin=oy;
	lons=np.linspace(lonmin,lonmin+lonstep*(nlon-1),nlon) #lonmin:lonstep:lonmin+lonstep*(nlon-1);
	lats=np.linspace(latmin,latmin+latstep*(nlat-1),nlat)
	[LONS,LATS]=np.meshgrid(lons,lats);
	
	grid_struct={'lons': lons, 'lats': lats, 
	'LONS': LONS, 'LATS': LATS,
	'nx': len(lons), 'ny': len(lats),
	'dx': lonstep, 'dy': latstep
	}
# 	print(grid_struct)
	G=np.zeros([npath,nx*ny]);
	
	for ipath in range(npath):
		path=paths[ipath];
		lons_M=path[:,0];
		lats_M=path[:,1];
		
# 		xyzM=geo2cartesian(lons_M,lats_M);
		
		[iA, iB, iC] = indexes_delaunay_triangle(grid_struct, lons_M, lats_M);
# 		print('[iA, iB, iC]',[iA, iB, iC])
		ilonlatA=[grid_struct['LONS'].flatten(order='F')[iA],grid_struct['LATS'].flatten(order='F')[iA]];
		ilonlatB=[grid_struct['LONS'].flatten(order='F')[iB],grid_struct['LATS'].flatten(order='F')[iB]];
		ilonlatC=[grid_struct['LONS'].flatten(order='F')[iC],grid_struct['LATS'].flatten(order='F')[iC]];
		
# 		xyzA=np.concatenate([ilonlatA[0],ilonlatA[1],np.zeros(len(iA))[:,np.newaxis]],axis=1)
# 		xyzB=np.concatenate([ilonlatB[0],ilonlatB[1],np.zeros(len(iB))[:,np.newaxis]],axis=1)
# 		xyzC=np.concatenate([ilonlatC[0],ilonlatC[1],np.zeros(len(iC))[:,np.newaxis]],axis=1)
		xyzA=np.concatenate([ilonlatA[1],ilonlatA[0],np.zeros(len(iA))[:,np.newaxis]],axis=1)
		xyzB=np.concatenate([ilonlatB[1],ilonlatB[0],np.zeros(len(iB))[:,np.newaxis]],axis=1)
		xyzC=np.concatenate([ilonlatC[1],ilonlatC[0],np.zeros(len(iC))[:,np.newaxis]],axis=1)

# 		print(lons_M.shape,lats_M.shape,np.zeros(lats_M.shape[0])[:,np.newaxis].shape)
		xyzM=np.concatenate([lons_M[:,np.newaxis],lats_M[:,np.newaxis],np.zeros(lats_M.shape[0])[:,np.newaxis]],axis=1)
		
# 		xyzA=geo2cartesian(ilonlatA(:,1),ilonlatA(:,2));
# 		xyzB=geo2cartesian(ilonlatB(:,1),ilonlatB(:,2));
# 		xyzC=geo2cartesian(ilonlatC(:,1),ilonlatC(:,2));
		
		xyzMp = projection(xyzM, xyzA, xyzB, xyzC);
		print('xyzM',xyzM)
		print('xyzMp',xyzMp)
		print(' xyzA', xyzA)
		print(' xyzB', xyzB)
		print(' xyzC', xyzC)
		
		[wA, wB, wC] = barycentric_coords(xyzMp, xyzA, xyzB, xyzC);
# 		print('[wA wB wC]',[wA, wB, wC])
		
# 		print('[wA, wB, wC]',[wA, wB, wC])
		#attributing weights to grid nodes along path:
		#w[j, :] = w_j(r) = weights of node j along path
		nM = path.shape[0];
		w = np.zeros([nx*ny,nM]);
		for i in range(nM):
			w[iA[i], i] = wA[i];
			w[iB[i], i] = wB[i];
			w[iC[i], i] = wC[i];
	
		print('wA,wB,wC',wA,wB,wC)
# 		ds =  np.sqrt((lats_M[0:-2]-lats_M[1:-1])*(lats_M[0:-2]-lats_M[1:-1])+(lons_M[0:-2]-lons_M[1:-1])*(lons_M[0:-2]-lons_M[1:-1]));
		ds =  np.sqrt((lats_M[0:-1]-lats_M[1:])*(lats_M[0:-1]-lats_M[1:])+(lons_M[0:-1]-lons_M[1:])*(lons_M[0:-1]-lons_M[1:]));

		print('w',w)
		print('w.shape',w.shape)
		
		
		print('ds.shape',ds.shape)
# 		print('ds',ds)
		print('nM',nM)
		print('(w[:,0:-2] + w[:,2:-1]).shape',(w[:,0:-2] + w[:,1:-1]).shape)
# 		print('np.multiply((w[:,0:-2] + w[:,1:-1]), ds[:,np.newaxis]).shape',np.matmul((w[:,0:-2] + w[:,1:-1]), ds[:,np.newaxis]).shape)
# 		G[ipath, :] = 0.5 * np.matmul((w[:,0:-2] + w[:,1:-1]), ds[:,np.newaxis]).flatten();
# 		G[ipath, :] = 0.5 * np.matmul((w[:,0:-1] + w[:,1:]), ds[:,np.newaxis]).flatten();
		G[ipath, :] = np.matmul((w[:,0:-1]), ds[:,np.newaxis]).flatten();
	return G
	
def projection(xyzM, xyzA, xyzB, xyzC):
	'''
	projection: projection of M on the plane ABC
	Orthogonal projection of point(s) M on plane(s) ABC.
	Each point (M, A, B, C) should be a tuple of floats or a tuple of arrays, (x, y, z)
	Verified to be correct.
	'''
	M=xyzM;
	A=xyzA;
	B=xyzB;
	C=xyzC;

	AB = B-A;
	AC = C-A;
	MA = A-M;
	
	u = np.cross(AB,AC)
	norm_u=np.sqrt(np.sum(u*u,axis=1))
	MMp=np.zeros(u.shape);
	
	for n in range(norm_u.shape[0]):
# 		u[n,:] = u[n,:] / (norm_u[n]+0.000000001);
		u[n,:] = u[n,:] / (norm_u[n]);
		MA_dot_u=np.sum(MA[n,:]*u[n,:]);
		MMp[n,:]=MA_dot_u*u[n,:];
	
	xyzMp=MMp+M;
	
	return xyzMp
	

def barycentric_coords(xyzMp, xyzA, xyzB, xyzC):
	'''
	Barycentric coordinates of point(s) M in triangle(s) ABC.
	Each point (M, A, B, C) should be a tuple of floats or
	a tuple of arrays, (x, y, z).
	Barycentric coordinate wrt A (resp. B, C) is the relative
	area of triangle MBC (resp. MAC, MAB).
	Verified to be correct.
	'''
	
	M=xyzMp;
	A=xyzA;
	B=xyzB;
	C=xyzC;

	MA = A-M;
	MB = B-M;
	MC = C-M;

	#area of triangle = norm of vectorial product / 2
	wA = np.sqrt(np.sum(np.cross(MB, MC)*np.cross(MB, MC),axis=1)) / 2.0;
	wB = np.sqrt(np.sum(np.cross(MA, MC)*np.cross(MA, MC),axis=1)) / 2.0;
	wC = np.sqrt(np.sum(np.cross(MA, MB)*np.cross(MA, MB),axis=1)) / 2.0;
	wtot = wA + wB + wC;

# 	wA=wA / (wtot+0.00000001);
# 	wB=wB / (wtot+0.00000001);
# 	wC=wC / (wtot+0.00000001);

	wA=wA / wtot;
	wB=wB / wtot;
	wC=wC / wtot;
	
# 	if wA == wB and wB==wC:
# 		wA=1/3.0;
# 		wB=1/3.0;
# 		wC=1/3.0;

# 	wA[wA ==0 ]=1
# 	wB[wB ==0 ]=1
# 	wC[wC ==0 ]=1
	
	return wA, wB, wC


def indexes_delaunay_triangle(grid_struct, x, y):
	'''
	
	Indexes of the grid's nodes defining the
	Delaunay triangle around point (x, y)
	x and y indexes of bottom left neighbour
	
	'''
	
	xs=grid_struct['lons'];
	ys=grid_struct['lats'];
	
	# force to be column vector
	if xs.ndim==2:
		if xs.shape[1] > xs.shape[0]:
			xs=xs.flatten();
			ys=ys.flatten();

	dx=grid_struct['dx'];
	dy=grid_struct['dy'];
	nx=grid_struct['nx'];
	ny=grid_struct['ny'];
	
	ix=np.floor((x-np.min(xs))/dx);
	iy=np.floor((y-np.min(ys))/dy);
	
	
	ix=ix.astype('int')
	iy=iy.astype('int')
	
	print('ix,iy',ix,iy)
	xratio = (x - xs[ix]) / dx; 
	yratio = (y - ys[iy]) / dy;

	ix[xratio==0]=ix[xratio==0]-1;
	iy[yratio==0]=iy[yratio==0]-1;

	
	# returning indexes of vertices of bottom right triangle
	# or upper left triangle depending on location
	index1 = np.zeros([len(xratio),1],dtype='int')
	index2 = np.zeros([len(xratio),1],dtype='int')
	index3 = np.zeros([len(xratio),1],dtype='int')
	
# 	for n in range(len(xratio)):
# 		if xratio[n]==0 and yratio[n]==0:
# 			index1[n,0] = (iy[n]+1)*nx+ix[n]+1;
# 		elif xratio[n]==0 and yratio[n]!=0:
# 			index1[n,0] = (iy[n])*nx+ix[n]+1;
# 		elif xratio[n]!=0 and yratio[n]==0:
# 			index1[n,0] = (iy[n]+1)*nx+ix[n];
# 		else:
# 			index1[n,0] = (iy[n])*nx+ix[n];
# 	
# 	for n in range(len(xratio)):
# 		if xratio[n]>=yratio[n]:
# 			if yratio[n]!=0:
# 				index2[n,0]=(iy[n])*ny+ix[n]+1;
# 			else:
# 				index2[n,0]=(iy[n]+1)*ny+ix[n]+1;
# 		else:
# 			if xratio[n]!=0:
# 				index2[n,0]=(iy[n]+1)*ny+ix[n];
# 			else:
# 				index2[n,0]=(iy[n]+1)*ny+ix[n]+1;

# using backward grids
	index1[:,0] = iy*nx+ix;
	for n in range(len(xratio)):			
		if xratio[n]>=yratio[n]:
			index2[n,0]=(iy[n])*nx+ix[n]+1;
		else:
			index2[n,0]=(iy[n]+1)*nx+ix[n];

# 	index2[:,0] = index2[:,0];
	index3[:,0] = (iy+1)*nx+ix+1;

# using forward grids	
# 	index1[:,0] = (iy+1)*nx+ix+1;
# 	for n in range(len(xratio)):			
# 		if xratio[n]>=yratio[n]:
# 			if ix[n]+2>=nx:
# 				index2[n,0]=(iy[n]+1)*nx+ix[n]+1;
# 			else:
# 				index2[n,0]=(iy[n]+1)*nx+ix[n]+2;
# 		else:
# 			if iy[n]+2>=ny:
# 				index2[n,0]=(iy[n]+1)*nx+ix[n]+1;
# 			else:
# 				index2[n,0]=(iy[n]+2)*nx+ix[n]+1;

# 	index2[:,0] = index2[:,0];
# 
# 	for n in range(len(xratio)):	
# 		if iy[n]+2>=ny:
# 			iiy=iy[n]+1
# 		else:
# 			iiy=iy[n]+2
# 			
# 		if ix[n]+2>=nx:
# 			iix=ix[n]+1
# 		else:
# 			iix=ix[n]+2
# 			
# 		index3[n,0]=iiy*nx+iix;
	
# 	ix[ix==nx-3]=nx-3
# 	iy[iy==ny-3]=ny-3
# 	index3[:,0] = (iy+2)*nx+ix+2;
	
	return index1,index2,index3
	



	