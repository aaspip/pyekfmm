# This is a DEMO script to benchmark between Pyefkmm and bh_tomo (Matlab) packages
#
# By Yangkang Chen, Feb 24, 2025
#
# First download the benchmark data output from bh_tomo package
# https://github.com/aaspip/data/blob/main/bench_raytracing2d.mat

import scipy
data=scipy.io.loadmat('bench_raytracing2d.mat')
ox=float(data['g'][0][0][0]);
oz=float(data['g'][0][0][1]);
dx=float(data['g'][0][0][2]);
dz=float(data['g'][0][0][3]);
nx=int(data['g'][0][0][4]);
nz=int(data['g'][0][0][5]);
ox=0;
oz=0;
dx=0.2;
dz=0.2;
nx=41;
nz=41;


slowness=data['s']
vel=1.0/slowness.reshape([nz,nx],order='F')

rays=data['rays']

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
plt.subplot(121)

plt.imshow(vel,cmap=plt.cm.jet, interpolation='none', extent=[0,dx*(nx-1),dz*(nz-1),0]);
plt.xlabel('Lateral (km)');plt.ylabel('Depth (km)');
plt.colorbar(orientation='horizontal',shrink=0.6,label='Velocity (km/s)');

#put receiver
for ii in data['Rx']:
	plt.plot(ii[0],ii[1],marker='v',markersize=12,color='b')
#put 2D rays
for ii in range(len(rays)):
	plt.plot(rays[ii][0][:,0],rays[ii][0][:,1],color='k')
#put source
for ii in data['Tx']:
	plt.plot(0,0,marker='*',markersize=12,color='red')
plt.title('bh_tomo package')

##########################################################################################
## Below is for pyekfmm
##########################################################################################
import numpy as np
import pyekfmm as fmm

vxyz=np.transpose(vel, (1,0));
sx=0;
sy=0;
sz=0;
dy=dx;ny=1;

t=fmm.eikonal(vel.transpose().flatten(order='F'),xyz=np.array([sx,sy,sz]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,nz,order='F');#first axis (vertical) is x, second is y, third is z
tzx=np.transpose(time, (1,0));
rays2=[]
for ii in data['Rx']:
	rx=float(ii[0])
	rz=float(ii[1])
	paths=fmm.ray2d(time,source=[0,0],receiver=[rx,rz],step=0.01,trim=1,ax=[0,dx,nx],ay=[0,dz,nz])#,trim=0.01
	rays2.append(paths.transpose())


# import matplotlib.pyplot as plt
plt.subplot(122)
plt.imshow(vel,cmap=plt.cm.jet, interpolation='none', extent=[0,dx*(nx-1),dz*(nz-1),0]);
plt.xlabel('Lateral (km)');plt.ylabel('Depth (km)');
plt.colorbar(orientation='horizontal',shrink=0.6,label='Velocity (km/s)');

#put receiver
for ii in data['Rx']:
	plt.plot(ii[0],ii[1],marker='v',markersize=12,color='b')
#put 2D rays
for ii in range(len(rays2)):
	plt.plot(rays2[ii][:,0],rays2[ii][:,1],color='k')
#put source
for ii in data['Tx']:
	plt.plot(0,0,marker='*',markersize=12,color='red')
plt.title('Pyekfmm package')

plt.savefig('test_pyekfmm_raytracing2d_benchmarkWITHbhtomo.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)

plt.show()
	




