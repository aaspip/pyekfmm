import numpy as np

def stream2d(u,v, dx, dy, sx, sy, step=0.1, maxvert=10000):
	"""
	stream2d: draw 2D stream lines along the steepest descent direction
	
	INPUT
	u:   	derivative of traveltime in x
	v:   	derivative of traveltime in z
	sx:			source relative coordinate in x [e.g., n.a, n is integer (grid NO) and a is floating number]
	sy:			source relative coordinate in y  [e.g., n.a, n is integer (grid NO) and a is floating number]
	step:	ray increment length
	maxvert: maximum number of verts
	
	OUTPUT  
	verts: verts locations in [x,y,z]
	numverts: number of verts
	
	Copyright (C) 2023 The University of Texas at Austin
	Copyright (C) 2023 Yangkang Chen
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	
	References   
	[1] Chen et al., 2023, Pyekfmm: a python package for 3D fast-marching-based traveltime calculation and its applications in seismology, SRL.
		
	 DEMO
	 demos/test_xxx.py
	"""
	
	xSize=u.shape[0]
	ySize=u.shape[1]
	
	[verts, numverts] = traceStreamUV  (u.flatten(order='F'),v.flatten(order='F'),xSize, ySize, dx, dy, sx, sy, step, maxvert);
	verts=verts.reshape(2,numverts,order='F');
	
	return verts,numverts

def stream3d(u,v, w, dx, dy, dz, sx, sy, sz, step=0.1, maxvert=10000):
	"""
	stream3d: draw 3D stream lines along the steepest descent direction
	
	INPUT
	u:   	derivative of traveltime in x
	v:   	derivative of traveltime in z
	w:   	derivative of traveltime in y
	sx:			source relative coordinate in x [e.g., n.a, n is integer (grid NO) and a is floating number]
	sy:			source relative coordinate in y  [e.g., n.a, n is integer (grid NO) and a is floating number]
	sz:			source relative coordinate in z  [e.g., n.a, n is integer (grid NO) and a is floating number]
	step:	ray increment length
	maxvert: maximum number of verts
	
	OUTPUT  
	verts: verts locations in [x,y,z]
	numverts: number of verts
	
	Copyright (C) 2023 The University of Texas at Austin
	Copyright (C) 2023 Yangkang Chen
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	
	References   
	[1] Chen et al., 2023, Pyekfmm: a python package for 3D fast-marching-based traveltime calculation and its applications in seismology, SRL.
		
	 DEMO
	 demos/test_xxx.py
	"""
	
	xSize=u.shape[0]
	ySize=u.shape[1]
	zSize=u.shape[2]
	[verts, numverts] = traceStreamUVW  (u.flatten(order='F'),v.flatten(order='F'),w.flatten(order='F'), xSize, ySize, zSize, dx, dy, dz, sx, sy, sz, step, maxvert);
	verts=verts.reshape(3,numverts,order='F');
	
	return verts,numverts
	
def traceStreamUV (ugrid, vgrid, xdim, ydim, dx, dy, sx, sy, step, maxvert):
	"""
	traceStreamUV: 2D streamline
	
	INPUT
	ugrid:   	derivative of traveltime in x
	vgrid:   	derivative of traveltime in y 
	xdim:		number of samples in x
	ydim:		number of samples in y
	sx:			source relative coordinate in x [e.g., n.a, n is integer (grid NO) and a is floating number]
	sy:			source relative coordinate in y  [e.g., n.a, n is integer (grid NO) and a is floating number]
	step:	ray increment length
	maxvert: maximum number of verts
	
	OUTPUT  
	verts: verts locations in [x,y]
	numverts: number of verts
	
	Copyright (C) 2023 The University of Texas at Austin
	Copyright (C) 2023 Yangkang Chen
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	
	References   
	[1] Chen et al., 2023, Pyekfmm: a python package for 3D fast-marching-based traveltime calculation and its applications in seismology, SRL.
		
	 DEMO
	 demos/test_xxx.py
	"""
	numverts=0;
	x=sx-1;y=sy-1;
	verts=np.zeros([2*maxvert,1])
	while 1:
		if x<0:
			x=0;
		if y<0:
			y=0;
			
		if x>=xdim-1:
			x=xdim-1;
		if y>=ydim-1:
			y=ydim-1;
	
		if (x<0 or x>xdim or y<0 or y>ydim or numverts>=maxvert) : 
# 			print("First break");
			break;
		
		ix=int(np.floor(x))
		iy=int(np.floor(y))
		
		if ix == xdim-1:
			ix=ix-1;
			
		if iy == ydim-1:
			iy=iy-1;
			
		xfrac=x-ix;
		yfrac=y-iy;
		
		#weights for linear interpolation
		a=(1-xfrac)*(1-yfrac);
		b=(  xfrac)*(1-yfrac);
		c=(1-xfrac)*(  yfrac);
		d=(  xfrac)*(  yfrac);
		
		verts[2*numverts + 0] = x+1;
		verts[2*numverts + 1] = y+1;
		
		#if already been here, done
		if numverts>=2:
			if verts[2*numverts] == verts[2*(numverts-2)] and verts[2*numverts+1] == verts[2*(numverts-2)+1]:
				numverts=numverts+1;
# 				print("Second break");
				break;
		
		numverts=numverts+1;
		ui = ugrid[ix  +xdim*iy]*a + ugrid[ix  +xdim*(iy+1)]*b + ugrid[ix+1+xdim*iy]*c + ugrid[ix+1+xdim*(iy+1)]*d;
		vi = vgrid[ix  +xdim*iy]*a + vgrid[ix  +xdim*(iy+1)]*b + vgrid[ix+1+xdim*iy]*c + vgrid[ix+1+xdim*(iy+1)]*d;
		
		#calculate step size, if 0, done
		if abs(ui) > abs(vi):
			imax=abs(ui);
		else:
			imax=abs(vi);
			
		if imax==0:
			print("Third break");
			break;
		
		imax=step/imax;
		ui = ui*imax;
		vi = vi*imax;
		
		#update the current position
		x = x+ui/dx;
		y = y+vi/dy;
	
# 	print('numverts',numverts)
	verts=verts[0:2*numverts]
	
	return verts,numverts

def traceStreamUVW (ugrid, vgrid, wgrid, xdim, ydim,  zdim, dx, dy, dz, sx, sy, sz, step, maxvert):
	"""
	traceStreamUVW: 3D streamline
	
	INPUT
	ugrid:   	derivative of traveltime in x
	vgrid:   	derivative of traveltime in y
	wgrid:   	derivative of traveltime in z
	xdim:		number of samples in x
	ydim:		number of samples in y
	zdim:		number of samples in z
	dx:			sampling in x
	dy:			sampling in y
	dz:			sampling in z
	sx:			source relative coordinate in x [e.g., n.a, n is integer (grid NO) and a is floating number]
	sy:			source relative coordinate in y  [e.g., n.a, n is integer (grid NO) and a is floating number]
	sz:			source relative coordinate in z  [e.g., n.a, n is integer (grid NO) and a is floating number]
	step:	ray increment length
	maxvert: maximum number of verts
	
	OUTPUT  
	verts: verts locations in [x,y,z]
	numverts: number of verts
	
	Copyright (C) 2023 The University of Texas at Austin
	Copyright (C) 2023 Yangkang Chen
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details: http://www.gnu.org/licenses/
	
	References   
	[1] Chen et al., 2023, Pyekfmm: a python package for 3D fast-marching-based traveltime calculation and its applications in seismology, SRL.
		
	 DEMO
	 demos/test_xxx.py
	"""
	
	numverts=0;
	
	x=sx-1.0;y=sy-1.0;z=sz-1.0;
	verts=np.zeros([3*maxvert,1])

	while 1:
		if x<0:
			x=0;
		if y<0:
			y=0;
		if z<0:
			z=0;
			
		if x>=xdim-1:
			x=xdim-1;
		if y>=ydim-1:
			y=ydim-1;
		if z>=zdim-1:
			z=zdim-1;
				
		if (x<0 or x>xdim or y<0 or y>ydim or z<0 or z>zdim or numverts>=maxvert) :
			print("First break");
			break;

		ix=int(np.floor(x))
		iy=int(np.floor(y))
		iz=int(np.floor(z))
		
		if ix == xdim-1:
			ix=ix-1;
			
		if iy == ydim-1:
			iy=iy-1;

		if iz == zdim-1:
			iz=iz-1;
		
		xfrac=x-ix;
		yfrac=y-iy;
		zfrac=z-iz;
		#weights for linear interpolation
# 		a=(1-xfrac)*(1-yfrac)*(1-zfrac);
# 		b=(  xfrac)*(1-yfrac)*(1-zfrac);
# 		c=(1-xfrac)*(  yfrac)*(1-zfrac);
# 		d=(  xfrac)*(  yfrac)*(1-zfrac);
# 		e=(1-xfrac)*(1-yfrac)*(  zfrac);
# 		f=(  xfrac)*(1-yfrac)*(  zfrac);
# 		g=(1-xfrac)*(  yfrac)*(  zfrac);
# 		h=(  xfrac)*(  yfrac)*(  zfrac);

		a=(1-xfrac)*(1-yfrac)*(1-zfrac);
		b=(1-xfrac)*(  yfrac)*(1-zfrac);
		c=(  xfrac)*(1-yfrac)*(1-zfrac);
		d=(  xfrac)*(  yfrac)*(1-zfrac);
		e=(1-xfrac)*(1-yfrac)*(  zfrac);
		f=(1-xfrac)*(  yfrac)*(  zfrac);
		g=(  xfrac)*(1-yfrac)*(  zfrac);
		h=(  xfrac)*(  yfrac)*(  zfrac);
	
		verts[3*numverts + 0] = x+1;
		verts[3*numverts + 1] = y+1;
		verts[3*numverts + 2] = z+1;
		#if already been here, done
		if numverts>=2:
			if verts[3*numverts] == verts[3*(numverts-2)] and verts[3*numverts+1] == verts[3*(numverts-2)+1] and verts[3*numverts+2] == verts[3*(numverts-2)+2]:
				numverts=numverts+1;
				print("Second break");
				break;
		
		numverts=numverts+1;
# 		ui = ugrid[iy  +ydim*ix +xdim*ydim*iz]*a + ugrid[iy  +ydim*(ix+1) +xdim*ydim*iz]*b + ugrid[iy+1+ydim*ix+xdim*ydim*iz]*c + ugrid[iy+1+ydim*(ix+1)+xdim*ydim*iz]*d + ugrid[iy  +ydim*ix +xdim*ydim*(iz+1)]*e + ugrid[iy  +ydim*(ix+1) +xdim*ydim*(iz+1)]*f + ugrid[iy+1+ydim*ix+xdim*ydim*(iz+1)]*g + ugrid[iy+1+ydim*(ix+1)+xdim*ydim*(iz+1)]*h;
# 		vi = vgrid[iy  +ydim*ix +xdim*ydim*iz]*a + vgrid[iy  +ydim*(ix+1) +xdim*ydim*iz]*b + vgrid[iy+1+ydim*ix+xdim*ydim*iz]*c + vgrid[iy+1+ydim*(ix+1)+xdim*ydim*iz]*d + vgrid[iy  +ydim*ix +xdim*ydim*(iz+1)]*e + vgrid[iy  +ydim*(ix+1) +xdim*ydim*(iz+1)]*f + vgrid[iy+1+ydim*ix+xdim*ydim*(iz+1)]*g + vgrid[iy+1+ydim*(ix+1)+xdim*ydim*(iz+1)]*h;
# 		wi = wgrid[iy  +ydim*ix +xdim*ydim*iz]*a + wgrid[iy  +ydim*(ix+1) +xdim*ydim*iz]*b + wgrid[iy+1+ydim*ix+xdim*ydim*iz]*c + wgrid[iy+1+ydim*(ix+1)+xdim*ydim*iz]*d + wgrid[iy  +ydim*ix +xdim*ydim*(iz+1)]*e + wgrid[iy  +ydim*(ix+1) +xdim*ydim*(iz+1)]*f + wgrid[iy+1+ydim*ix+xdim*ydim*(iz+1)]*g + wgrid[iy+1+ydim*(ix+1)+xdim*ydim*(iz+1)]*h;

# 		ui = ugrid[ix  +xdim*iy +xdim*ydim*iz]*a + ugrid[ix  +xdim*(iy+1) +xdim*ydim*iz]*b + ugrid[ix+1+xdim*iy+xdim*ydim*iz]*c + ugrid[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + ugrid[ix  +xdim*iy +xdim*ydim*(iz+1)]*e + ugrid[ix  +xdim*(iy+1) +xdim*ydim*(iz+1)]*f + ugrid[ix+1+xdim*iy+xdim*ydim*(iz+1)]*g + ugrid[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;
# 		vi = vgrid[ix  +xdim*iy +xdim*ydim*iz]*a + vgrid[ix  +xdim*(iy+1) +xdim*ydim*iz]*b + vgrid[ix+1+xdim*iy+xdim*ydim*iz]*c + vgrid[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + vgrid[ix  +xdim*iy +xdim*ydim*(iz+1)]*e + vgrid[ix  +xdim*(iy+1) +xdim*ydim*(iz+1)]*f + vgrid[ix+1+xdim*iy+xdim*ydim*(iz+1)]*g + vgrid[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;
# 		wi = wgrid[ix  +xdim*iy +xdim*ydim*iz]*a + wgrid[ix  +xdim*(iy+1) +xdim*ydim*iz]*b + wgrid[ix+1+xdim*iy+xdim*ydim*iz]*c + wgrid[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + wgrid[ix  +xdim*iy +xdim*ydim*(iz+1)]*e + wgrid[ix  +xdim*(iy+1) +xdim*ydim*(iz+1)]*f + wgrid[ix+1+xdim*iy+xdim*ydim*(iz+1)]*g + wgrid[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;

		ui = ugrid[ix  +xdim*iy    +xdim*ydim*iz]*a + \
			 ugrid[ix  +xdim*(iy+1)+xdim*ydim*iz]*b + \
			 ugrid[ix+1+xdim*iy    +xdim*ydim*iz]*c + \
			 ugrid[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + \
			 ugrid[ix  +xdim*iy    +xdim*ydim*(iz+1)]*e + \
			 ugrid[ix  +xdim*(iy+1)+xdim*ydim*(iz+1)]*f + \
			 ugrid[ix+1+xdim*iy    +xdim*ydim*(iz+1)]*g + \
			 ugrid[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;

		vi = vgrid[ix  +xdim*iy    +xdim*ydim*iz]*a + \
			 vgrid[ix  +xdim*(iy+1)+xdim*ydim*iz]*b + \
			 vgrid[ix+1+xdim*iy    +xdim*ydim*iz]*c + \
			 vgrid[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + \
			 vgrid[ix  +xdim*iy    +xdim*ydim*(iz+1)]*e + \
			 vgrid[ix  +xdim*(iy+1)+xdim*ydim*(iz+1)]*f + \
			 vgrid[ix+1+xdim*iy    +xdim*ydim*(iz+1)]*g + \
			 vgrid[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;

		wi = wgrid[ix  +xdim*iy    +xdim*ydim*iz]*a + \
			 wgrid[ix  +xdim*(iy+1)+xdim*ydim*iz]*b + \
			 wgrid[ix+1+xdim*iy    +xdim*ydim*iz]*c + \
			 wgrid[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + \
			 wgrid[ix  +xdim*iy    +xdim*ydim*(iz+1)]*e + \
			 wgrid[ix  +xdim*(iy+1)+xdim*ydim*(iz+1)]*f + \
			 wgrid[ix+1+xdim*iy    +xdim*ydim*(iz+1)]*g + \
			 wgrid[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;
			 
		#calculate step size, if 0, done
		if abs(ui) > abs(vi):
			imax=abs(ui);
		else:
			imax=abs(vi);
			
		if abs(wi)>imax:
			imax=abs(wi);

		if imax==0:
			print("Third break");
			break;

		imax=step/imax;
		ui = ui*imax;
		vi = vi*imax;
		wi = wi*imax;
		
		#update the current position
		x = x+ui/dx;
		y = y+vi/dy;
		z = z+wi/dz; ##very tricky solution for dx!=dy!=dz case

	verts=verts[0:3*numverts] 
	
	return verts,numverts
	
	


def trimrays(paths, start_points, T=None):
	"""
	trimrays: trim rays (remove very close ray points around the source)
	
	INPUT
	%paths: [2 x ngrid]
	%start_points: [2 x 1], e.g., start_points=np.array([1,1])
	%T: threshold, e.g., dx,dz
	
	OUTPUT
	paths: 	trimmed rays
	"""
	ngrid=paths.shape[1]
	start_points=np.expand_dims(start_points,1);
	
	d=np.sqrt(np.sum(np.power(paths-np.repeat(start_points,ngrid,axis=1),2),0));
	
	if T is None:
		T = d.max()/300/300;
		
	I=np.where(d<T)
	
	paths=np.delete(paths,I,1)
	
	paths=np.concatenate((paths,start_points),axis=1)
	
	return paths
	
	
def ray2d(time,source,receiver,ax=[0,0.01,101],ay=[0,0.01,101],step=0.1,maxvert=10000,**kws):
	'''
	ray2d: 2D ray tracing
	
	INPUT
	time: 2D traveltime
	source: 	source location [sx,sy]
	receiver: receiver location [rx,ry]
	ax=[0,dx,nx],ay=[0,dy,ny]
	step:ray segment increase
	maxvert: maximum number of verts
	
	kws:	other key words (e.g., trim=0.5)
	trim:  allowable distance between the source point and the second last ray point (in absolute grid spacing)
	trim = 0.000000001 means no trimming; when trim=0, the number of ray points will increase by 1, adding the source.
	
	OUTPUT
	paths: ray paths [x,y]
	'''
	
	## Gradient calculation
	dx=ax[1];dy=ay[1];
	nx=ax[2];ny=ay[2];

	if nx==1 and ny==1:
		Exception("INPUT ERROR: the input time must have at least two dimensions")
			
	time=np.squeeze(time)
		
	print('dx,dy',dx,dy)
	tx,ty = np.gradient(time)

	tx,ty=tx/dx,ty/dy
	
	receiverx=(receiver[0]-ax[0])/ax[1]+1
	receivery=(receiver[1]-ay[0])/ay[1]+1
	
	paths,nrays=stream2d(-tx,-ty, dx, dy, receiverx, receivery, step=step, maxvert=maxvert)

	print('Before trim',paths.shape)
	if 'trim' in kws.keys():
		sourcex=(source[0]-ax[0])/ax[1]+1
		sourcey=(source[1]-ay[0])/ay[1]+1
		## trim the rays and add the source point
		paths=trimrays(paths,start_points=np.array([sourcex,sourcey]),T=kws['trim']) #e.g., 0.5
	print('After trim',paths.shape)
		
	paths[0,:]=(paths[0,:]-1)*dx;
	paths[1,:]=(paths[1,:]-1)*dy;
	
	return paths	


def ray3d(time,source,receiver,ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],step=0.1,maxvert=10000,**kws):
	'''
	ray3d: 3D ray tracing
	
	INPUT
	time: 3D traveltime
	source: 	source location [sx,sy,sz]
	receiver: receiver location [rx,ry,rz]
	ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz]
	step:ray segment increase
	maxvert: maximum number of verts
	
	kws:	other key words (e.g., trim=0.5)
	trim:  allowable distance between the source point and the second last ray point (in absolute grid spacing)
	trim = 0.000000001 means no trimming; when trim=0, the number of ray points will increase by 1, adding the source.
	
	OUTPUT
	paths: ray paths [x,y,z]
	'''
	
	print('receiver',receiver)
	print('source',source)
	
	## Gradient calculation
	dx=ax[1];dy=ay[1];dz=az[1];
	nx=ax[2];ny=ay[2];nz=az[2];
	print('dx,dy,dz',dx,dy,dz)
	print('nx,ny,nz',nx,ny,nz)
	
	tx,ty,tz = np.gradient(time)

	tx,ty,tz=tx/dx,ty/dy,tz/dz
	
	receiverx=(receiver[0]-ax[0])/ax[1]+1
	receivery=(receiver[1]-ay[0])/ay[1]+1
	receiverz=(receiver[2]-az[0])/az[1]+1
	
	paths,nrays=stream3d(-tx,-ty, -tz, dx, dy, dz, receiverx, receivery, receiverz, step=step, maxvert=maxvert)

	print('Before trim',paths.shape)
	if 'trim' in kws.keys():
		sourcex=(source[0]-ax[0])/ax[1]+1
		sourcey=(source[1]-ay[0])/ay[1]+1
		sourcez=(source[2]-az[0])/az[1]+1
		
		## trim the rays and add the source point
		paths=trimrays(paths,start_points=np.array([sourcex,sourcey,sourcez]),T=kws['trim']) #e.g., 0.5
	print('After trim',paths.shape)

	paths[0,:]=(paths[0,:]-1)*dx;
	paths[1,:]=(paths[1,:]-1)*dy;
	paths[2,:]=(paths[2,:]-1)*dz;
	
	return paths	
	
def extract(time, point, ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101]):
	'''
	extract: extract values (time/dip/azimuth) from 3D array according to the point location [x,y,z] (absolute coordinates)
	
	INPUT
	time: 	3D traveltime/dip/azimuth [ix,iy,iz] 
	point: 	point location [x,y,z] for extracting time/dip/azim
	ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz]
	
	OUTPUT
	tpoint: time/dip/azim at point [x,y,z]
	
	EXAMPLE
	import pyekfmm as fmm
	import numpy as np

	vel=3.0*np.ones([101*101*101,1],dtype='float32');
	t=fmm.eikonal(vel,xyz=np.array([0.5,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2);
	time=t.reshape(101,101,101,order='F'); #[x,y,z]

	## Verify
	print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
	print(['Correct result:',0.49965078, 0.0, 0.08905013, 0.007929926])
	
	## Verify (First case on the edge)
	t1=fmm.extract(time,[0,0,0],ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101])
	print('Time at [0,0,0] from fmm.extract is:',t1)
	print('Time at [0,0,0] from time[0,0,0] is:',time[0,0,0])

	t1=fmm.extract(time,[1.0,0,0],ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101])
	print('Time at [1.0,0,0] from fmm.extract is:',t1)
	print('Time at [1.0,0,0] from time[100,0,0] is:',time[100,0,0])
	
	t1=fmm.extract(time,[1.0,1.0,1.0],ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101])
	print('Time at [1.0,1.0,1.0] from fmm.extract is:',t1)
	print('Time at [1.0,1.0,1.0] from time[100,100,100] is:',time[100,100,100])

	## Verify (Then cases between the nodes)
	t1=fmm.extract(time,[0.505,0,0],ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101])
	print('Time at [0.505,0,0] from fmm.extract is:',t1)
	print('Time at [0.505,0,0] analytically is:',0.005/3.0)

	t1=fmm.extract(time,[0.905,0,0],ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101])
	print('Time at [0.905,0,0] from fmm.extract is:',t1)
	print('Time at [0.905,0,0] analytically is:',0.405/3.0)
	
	
	#EXAMPLE2
	vel=3.0*np.ones([101*101*101,1],dtype='float32');
	t=fmm.eikonal(vel,xyz=np.array([0.505,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2);
	time=t.reshape(101,101,101,order='F'); #[x,y,z]

	print('Source off-the-grid at 0.505')
	print('Time at [0.5,0,0] from time[50,0,0] is:',time[50,0,0])
	print('Time at [0.5,0,0] from time[50,0,0] is:',0.005/3.0)
	'''

	xdim=time.shape[0]
	ydim=time.shape[1]
	zdim=time.shape[2]
	
	time=time.flatten(order='F');
	
	pointx=(point[0]-ax[0])/ax[1]
	pointy=(point[1]-ay[0])/ay[1]
	pointz=(point[2]-az[0])/az[1]
	
	ix=int(np.floor(pointx))
	iy=int(np.floor(pointy))
	iz=int(np.floor(pointz))
	
	if ix == xdim-1:
		ix=ix-1;
		
	if iy == ydim-1:
		iy=iy-1;

	if iz == zdim-1:
		iz=iz-1;
	
	xfrac=pointx-ix;
	yfrac=pointy-iy;
	zfrac=pointz-iz;
	
	#weights for linear interpolation
	a=(1-xfrac)*(1-yfrac)*(1-zfrac);
	b=(1-xfrac)*(  yfrac)*(1-zfrac);
	c=(  xfrac)*(1-yfrac)*(1-zfrac);
	d=(  xfrac)*(  yfrac)*(1-zfrac);
	e=(1-xfrac)*(1-yfrac)*(  zfrac);
	f=(1-xfrac)*(  yfrac)*(  zfrac);
	g=(  xfrac)*(1-yfrac)*(  zfrac);
	h=(  xfrac)*(  yfrac)*(  zfrac);

	tpoint = time[ix  +xdim*iy 	  +xdim*ydim*iz]*a + \
			 time[ix  +xdim*(iy+1)+xdim*ydim*iz]*b + \
			 time[ix+1+xdim*iy    +xdim*ydim*iz]*c + \
			 time[ix+1+xdim*(iy+1)+xdim*ydim*iz]*d + \
			 time[ix  +xdim*iy    +xdim*ydim*(iz+1)]*e + \
			 time[ix  +xdim*(iy+1)+xdim*ydim*(iz+1)]*f + \
			 time[ix+1+xdim*iy    +xdim*ydim*(iz+1)]*g + \
			 time[ix+1+xdim*(iy+1)+xdim*ydim*(iz+1)]*h;
	
	return tpoint
	
	
	
def extracts(time, points, ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101]):
	'''
	extracts: extract values (time/dip/azimuth) from 3D array according to the point locations [npoint x 3] (absolute coordinates)
	
	INPUT
	time: 	3D traveltime/dip/azimuth [ix,iy,iz] 
	points: many point locations [x,y,z] for extracting time/dip/azim
	ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz]
	
	OUTPUT
	tpoint: time/dip/azim at point [x,y,z]
	
	
	EXAMPLE
	import pyekfmm as fmm
	import numpy as np

	vel=3.0*np.ones([101*101*101,1],dtype='float32');
	t=fmm.eikonal(vel,xyz=np.array([0.5,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2);
	time=t.reshape(101,101,101,order='F'); #[x,y,z]

	## Verify
	print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
	print(['Correct result:',0.49965078, 0.0, 0.08905013, 0.007929926])

	t1=fmm.extracts(time,np.array([[0.0,0,0],[0.505,0,0]]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101])
	print('Time at [[0.0,0,0],[0.505,0,0]] from fmm.extract is:',t1)
	print('Time at [[0.0,0,0],[0.505,0,0]] analytically is:',0.5/3.0,0.005/3.0)
	
	'''
	if points.size==3:
		tpoint=extract(time, points, ax, ay, az)
	else:
		[n1,n2]=points.shape
		if n2 != 3:
			points=points.T #transpose
			[n1,n2]=points.shape
		
		tpoint=np.zeros(n1)
		for ii in range(n1):
			tpoint[ii]=extract(time, points[ii,:], ax, ay, az)
		
	return tpoint

		
	
	