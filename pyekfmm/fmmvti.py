def eikonalvti(velx,velz,eta,xyz,ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2,verb=1,angle=False):
	'''
	EIKONAL: Fast marching eikonal solver (3-D)
	
	INPUT
	velz: 1D numpy array (nx*ny*nz) (slower velocity)
	velx: 1D numpy array (nx*ny*nz) (faster velocity)
	eta: 1D numpy array (nx*ny*nz)
	xyz: 1D/2D numpy array (one event: 1x3 or multi-event: ne x 3)
	ax: axis x [ox,dx,nx]
	ay: axis y [oy,dy,ny]
	az: axis z [oz,dz,nz]
	order: accuracy order [1 or 2]
	
	OUTPUT
	times: traveltime in xyz respectively (1D numpy array)
		   (one event: nx*ny*nz or multi-event: nx*ny*nz*ne)
	
	EXAMPLE
	demos/test_first-fifth.py
	
	COPYRIGHT
	Yangkang Chen, 2022, The University of Texas at Austin
	
	MODIFICATIONS
	[1] By Yangkang Chen, Sep, 2022
	
	'''
	import numpy as np
	if xyz.size == 3:
		from eikonalvtic import eikonalvtic_oneshot
		x=xyz[0];y=xyz[1];z=xyz[2];
		if angle:
			
			from eikonalc import eikonalc_oneshot_angle
			n123=velz.size
			tmp=eikonalc_oneshot_angle(velz,x,y,z,ax[0],ay[0],az[0],ax[1],ay[1],az[1],ax[2],ay[2],az[2],order);
			times=tmp[0:n123]
			dips=tmp[n123:n123*2]
			azims=tmp[n123*2:]
		else:
			# Here because in the C cade, we treat the first vel as the slower velocity
			times=eikonalvtic_oneshot(velx,velz,eta,x,y,z,ax[0],ay[0],az[0],ax[1],ay[1],az[1],ax[2],ay[2],az[2],order);
	else:
		from eikonalc import eikonalc_multishots
		[ne,ndim]=xyz.shape;#ndim must be 3
		x=xyz[:,0];y=xyz[:,1];z=xyz[:,2];
		x=np.expand_dims(x,1);
		y=np.expand_dims(y,1);
		z=np.expand_dims(z,1);
		
		if angle:
			from eikonalc import eikonalc_multishots_angle
			n123=velz.size
			tmp=eikonalc_multishots_angle(velz,x,y,z,ax[0],ay[0],az[0],ax[1],ay[1],az[1],ax[2],ay[2],az[2],order,verb);
			times=tmp[0:n123*ne]
			dips=tmp[n123*ne:n123*ne*2]
			azims=tmp[n123*ne*2:]
			
		else:
			times=eikonalc_multishots(velz,x,y,z,ax[0],ay[0],az[0],ax[1],ay[1],az[1],ax[2],ay[2],az[2],order,verb);
	
	if angle:
		return times,dips,azims
	else:
		return times
	
	
def eikonalvti_surf(vel,xyz,ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2,verb=1):
	'''
	EIKONAL_SURF: Fast marching eikonal solver (3-D) and recording the traveltimes on the surface.
	
	INPUT
	vel: 1D numpy array (nx*ny*nz)
	xyz: 1D/2D numpy array (one event: 1x3 or multi-event: ne x 3)
	ax: axis x [ox,dx,nx]
	ay: axis y [oy,dy,ny]
	az: axis z [oz,dz,nz]
	order: accuracy order [1 or 2]
	
	OUTPUT
	times: traveltime in xy of nshots (1D numpy array)
		   size of times is nx*ny*nshots
	
	EXAMPLE
	demos/test_fourth3d.py
	
	COPYRIGHT
	Yangkang Chen, 2022, The University of Texas at Austin
	
	MODIFICATIONS
	[1] By Yangkang Chen, Sep, 2022
	'''
	from eikonalc import eikonalc_surf
	[ne,ndim]=xyz.shape;#ndim must be 3
	x=xyz[:,0];y=xyz[:,1];z=xyz[:,2];
	times=eikonalc_surf(vel,x,y,z,ax[0],ay[0],az[0],ax[1],ay[1],az[1],ax[2],ay[2],az[2],order,verb);
		
	return times
	
def eikonalvti_rtp(vel,rtp,ar=[0,0.01,101],at=[0,1.8,101],ap=[0,3.6,101],order=2,verb=1):
	'''
	EIKONAL_RTP: Fast marching eikonal solver (3-D) in spherical coordinates
	
	INPUT
	vel: 1D numpy array (nx*ny*nz)
	rtp: 1D/2D numpy array (one event: 1x3 or multi-event: ne x 3)
	ar: axis r [or,dr,nr]
	at: axis theta [ot,dt,nt]
	ap: axis phi [op,dp,np]
	order: accuracy order [1 or 2]
	
	OUTPUT
	times: traveltime in xyz respectively (1D numpy array)
		   (one event: nx*ny*nz or multi-event: nx*ny*nz*ne)
	
	EXAMPLE
	demos/test_first-fifth.py
	
	COPYRIGHT
	Yangkang Chen, 2022, The University of Texas at Austin
	
	MODIFICATIONS
	[1] By Yangkang Chen, Sep, 2022
	
	'''
	import numpy as np
	if rtp.size == 3:
		from eikonalc import eikonalc_oneshot_rtp
		r=rtp[0];t=rtp[1];p=rtp[2];
		times=eikonalc_oneshot_rtp(vel,r,t,p,ar[0],at[0],ap[0],ar[1],at[1],ap[1],ar[2],at[2],ap[2],order);
	else:
		from eikonalc import eikonalc_multishots_rtp
		[ne,ndim]=rtp.shape;#ndim must be 3
		r=rtp[:,0];t=rtp[:,1];p=rtp[:,2];
		r=np.expand_dims(r,1);
		t=np.expand_dims(t,1);
		p=np.expand_dims(p,1);
		times=eikonalc_multishots_rtp(vel,r,t,p,ar[0],at[0],ap[0],ar[1],at[1],ap[1],ar[2],at[2],ap[2],order,verb);
		
	return times
	
	