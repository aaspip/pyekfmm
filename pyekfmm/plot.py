from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def cseis():
	'''
	cseis: seismic colormap
	
	By Yangkang Chen
	June, 2022
	
	EXAMPLE
	from pyseistr import cseis
	import numpy as np
	from matplotlib import pyplot as plt
	plt.imshow(np.random.randn(100,100),cmap=cseis())
	plt.show()
	'''
	seis=np.concatenate(
(np.concatenate((0.5*np.ones([1,40]),np.expand_dims(np.linspace(0.5,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((0.25*np.ones([1,40]),np.expand_dims(np.linspace(0.25,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((np.zeros([1,40]),np.expand_dims(np.linspace(0,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose()),axis=1)

	return ListedColormap(seis)
	

def plot3d(d3d,frames=None,dz=0.01,dx=0.01,dy=0.01,z=None,x=None,y=None,nlevel=100,figname=None,showf=True,close=True,**kwargs):
	'''
	plot3d: plot beautiful 3D slices
	
	By Yangkang Chen
	June, 18, 2023
	
	EXAMPLE 1
	import numpy as np
	d3d=np.random.rand(100,100,100);
	from pyseistr import plot3d
	plot3d(d3d);
	
	EXAMPLE 2
	import scipy
	data=scipy.io.loadmat('/Users/chenyk/chenyk/matlibcyk/test/hyper3d.mat')['cmp']
	from pyseistr import plot3d
	plot3d(data);
	'''

	[nz,nx,ny] = d3d.shape;
	
	if frames is None:
		frames=[int(nz/2),int(nx/2),int(ny/2)]
		
	# X, Y, Z = np.meshgrid(np.arange(nx)*0.01, np.arange(ny)*0.01, np.arange(nz)*0.01)

	if z is None:
		z=np.arange(nz)*dz
	
	if x is None:
		x=np.arange(nx)*dx
		
	if y is None:
		y=np.arange(ny)*dy
	
	X, Y, Z = np.meshgrid(x, y, z)
	
	d3d=d3d.transpose([1,2,0])
	
	kw = {
    'vmin': d3d.min(),
    'vmax': d3d.max(),
    'levels': np.linspace(d3d.min(), d3d.max(), nlevel),
    'cmap':cseis()
	}
	kw.update(kwargs)
	
	if 'alpha' not in kw.keys():
		kw['alpha']=1.0
	
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, aspect='auto',projection='3d')
	plt.jet()

	# Plot contour surfaces
	_ = ax.contourf(
	X[:, :, -1], Y[:, :, -1], d3d[:, :, frames[0]].transpose(), #x,y,z
	zdir='z', offset=0, **kw
	)

	_ = ax.contourf(
	X[0, :, :], d3d[:, frames[2], :], Z[0, :, :],
	zdir='y', offset=0, **kw
	)
	C = ax.contourf(
	d3d[frames[1], :, :], Y[:, -1, :], Z[:, -1, :],
	zdir='x', offset=X.max(), **kw
	)

	plt.gca().set_xlabel("X",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z",fontsize='large', fontweight='normal')

	xmin, xmax = X.min(), X.max()
	ymin, ymax = Y.min(), Y.max()
	zmin, zmax = Z.min(), Z.max()
	ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
	plt.gca().invert_zaxis()

	# Colorbar
	if 'barlabel' in kw.keys():
		cbar=fig.colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format= "%.2f", label=kw['barlabel'])
		cbar.ax.locator_params(nbins=5)
		kwargs.__delitem__('barlabel')

	if figname is not None:
		kwargs.__delitem__('cmap')
# 		print(kwargs)
		plt.savefig(figname,**kwargs)
	
	if showf:
		plt.show()
	else:
		if close:
			plt.close() #or plt.clear() ?
		
def framebox(x1,x2,y1,y2,c=None,lw=None):
	'''
	framebox: for drawing a frame box
	
	By Yangkang Chen
	June, 2022
	
	INPUT
	x1,x2,y1,y2: intuitive
	
	EXAMPLE I
	from pyseistr.plot import framebox
	from pyseistr.synthetics import gensyn
	from matplotlib import pyplot as plt
	d=gensyn();
	plt.imshow(d);
	framebox(200,400,200,300);
	plt.show()

	EXAMPLE II
	from pyseistr.plot import framebox
	from pyseistr.synthetics import gensyn
	from matplotlib import pyplot as plt
	d=gensyn();
	plt.imshow(d);
	framebox(200,400,200,300,c='g',lw=4);
	plt.show()
	
	'''
	
	if c is None:
		c='r';
	if lw is None:
		lw=2;

	plt.plot([x1,x2],[y1,y1],linestyle='-',color=c,linewidth=lw);
	plt.plot([x1,x2],[y2,y2],linestyle='-',color=c,linewidth=lw);
	plt.plot([x1,x1],[y1,y2],linestyle='-',color=c,linewidth=lw);
	plt.plot([x2,x2],[y1,y2],linestyle='-',color=c,linewidth=lw);

	
	return
	
			
			
				