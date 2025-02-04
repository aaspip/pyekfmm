'''
This is an attempt of using pyekfmm to do 3D raytraing with
the goal of eventually determining takeoff angles.

This example creates a small model with a velocity gradient.
Velocities are slowest in the southwest corner at the surface,
and fastest at the northeast corner at the bottom.

This version compares results obtained by treating either
the receiver or earthquake as the source.

This example is written by Robert Skoumal and Yangkang Chen

'''

import numpy as np
import matplotlib.pyplot as plt
import pyekfmm as fmm
rng = np.random.default_rng(12345)

'''
Create a synthetic, example velocity model
'''
# Num cells in x,y,z directions
nx=25 # horizontal ('east-west')
ny=80 # horizontal ('north-south')
nz=72 # vertical

# Spacing, km
dx=1.
dy=1.
dz=0.5 # The vertical resolution will be often be finer than the horizontal

# Velocity gradients
x_gradient=0.05
y_gradient=0.07
z_gradient=0.10

########################################################################
## uncomment for finer grids (added by Yangkang)
# Make the grid finer
# nx=50 # horizontal ('east-west')
# ny=160 # horizontal ('north-south')
# nz=144 # vertical
# # Spacing, km
# dx=.5
# dy=.5
# dz=0.25 # The vertical resolution will be often be finer than the horizontal
# 
# x_gradient=0.05/2
# y_gradient=0.07/2
# z_gradient=0.10/2
########################################################################
## uncomment for finer grids (added by Yangkang)

# Determine x,y,z sample locations
x_km=np.arange(nx)*dx
y_km=np.arange(ny)*dy
z_km=np.arange(nz)*dz

vp_velocity_3d=3*np.ones([nx,ny,nz],dtype='float32')
for x in range(nx):
	vp_velocity_3d[x,:,:]+=(x*x_gradient)
for y in range(ny):
	vp_velocity_3d[:,y,:]+=(y*y_gradient)
for z in range(nz):
	vp_velocity_3d[:,:,z]+=(z*z_gradient)

########################################################################
## uncomment for constant velocity (added by Yangkang)
## when velocity is constant, abs(time)=0
########################################################################
## uncomment for constant velocity (added by Yangkang)
# vp_velocity_3d=3*np.ones(vp_velocity_3d.shape)
'''
Create sources and receivers. In reality, the number of sources >> number of receivers.
'''
# nreceivers=1
# nearthquakes=1
# earthquake_xyz=np.atleast_2d(np.asarray([17.94731 , 48.262173, 26.570248, 0, 0])) # x, y, z, horizontal uncertainty, vertical uncertainty
# receiver_xyz=np.atleast_2d(np.asarray([2.2, 9.3, 0. ])) # x, y, z
nreceivers=2
nearthquakes=2
earthquake_xyz=np.atleast_2d(np.asarray([[17.94731 , 48.262173, 26.570248, 0, 0],
										 [17.94731 , 48.262173, 26.570248, 0, 0]])) # x, y, z, horizontal uncertainty, vertical uncertainty
receiver_xyz=np.atleast_2d(np.asarray([[2.2, 9.3, 0. ],
									   [2.2, 9.3, 0. ]])) # x, y, z
########################################################################
## uncomment for regular grids (added by Yangkang)
# earthquake_xyz=np.atleast_2d(np.asarray([[17 , 48, 26, 0, 0],
# 										 [17 , 48, 26, 0, 0]])) # x, y, z, horizontal uncertainty, 
# receiver_xyz=np.atleast_2d(np.asarray([[2., 9., 0. ],
# 									   [2., 9., 0. ]])) # x, y, z
# uncomment for regular grids (added by Yangkang)
########################################################################
								   
'''
Uses pyekfmm to determine the traveltime, takeoff, and azimuth.
Treats the **RECEIVER** as the source
'''
eq_time=np.zeros((nreceivers,nearthquakes))
eq_dips=np.zeros((nreceivers,nearthquakes))
eq_azims=np.zeros((nreceivers,nearthquakes))

for ii in range(len(receiver_xyz)):
	print('Station: {}/{}'.format(ii,len(receiver_xyz)-1))
	t,d,a=fmm.eikonal(vp_velocity_3d.flatten(order='F'),xyz=receiver_xyz[ii,:],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2,angle=True)

	time=t.reshape(nx,ny,nz,order='F')
	dip=d.reshape(nx,ny,nz,order='F')
	azim=a.reshape(nx,ny,nz,order='F')

	tmp_earthquake_xyz=earthquake_xyz[:,:3] # Selects the first 3 columns that correspond to the modified earthquake location
	eq_time[ii,:]=fmm.extracts(time,tmp_earthquake_xyz,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	eq_dips[ii,:]=fmm.extracts(dip,tmp_earthquake_xyz,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	eq_azims[ii,:]=fmm.extracts(azim,tmp_earthquake_xyz,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])


'''
Uses pyekfmm to determine the traveltime, takeoff, and azimuth.
Treats the **EARTHQUAKE** as the source
'''
eq_time2=np.zeros((nreceivers,nearthquakes))
eq_dips2=np.zeros((nreceivers,nearthquakes))
eq_azims2=np.zeros((nreceivers,nearthquakes))

for ii in range(len(earthquake_xyz)):
	print('Earthquake: {}/{}'.format(ii,len(earthquake_xyz)-1))
	t,d,a=fmm.eikonal(vp_velocity_3d.flatten(order='F'),xyz=earthquake_xyz[ii,:3],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2,angle=True)

	time=t.reshape(nx,ny,nz,order='F')
	dip=d.reshape(nx,ny,nz,order='F')
	azim=a.reshape(nx,ny,nz,order='F')

	eq_time2[:,ii]=fmm.extracts(time,receiver_xyz,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	eq_dips2[:,ii]=fmm.extracts(dip,receiver_xyz,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	eq_azims2[:,ii]=fmm.extracts(azim,receiver_xyz,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])

print('Avg abs travel time difference: {} sec'.format(np.round(np.mean(np.abs(eq_time-eq_time2)),5)))

print('eq_time',eq_time)
print('eq_time2',eq_time2)
'''
Finds the largest travel time difference
'''
from pyekfmm import plot3d
from pyekfmm import ray3d


# # Determines the receiver,earthquake indicies of the largest travel time difference
ind_receiver,ind_earthquake=np.unravel_index(np.argmax(np.abs((eq_time-eq_time2))),eq_time.shape)

## by Revised by Yangkang (NOTE, there is still a tiny difference between the two rays, that is caused by trivial factors like velocity model, grid spacing, linear interpolation, etc., in certain cases, they should be almost the same, e.g., constant velocity and regular grids)

## Plots raypath considering the EARTHQUAKE as the source
t,d,a=fmm.eikonal(vp_velocity_3d.flatten(order='F'),xyz=earthquake_xyz[ind_receiver,:3],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2,angle=True)
time=t.reshape(nx,ny,nz,order='F')
time_zxy=np.transpose(time,(2,0,1))
plot3d(time_zxy,x=x_km,y=y_km,z=z_km,frames=[0,nx-1,0],cmap=plt.cm.jet,barlabel='Traveltime (s)',showf=False,close=False,alpha=0.7)
paths1=ray3d(time,source=earthquake_xyz[ind_receiver,:3],receiver=receiver_xyz[ind_receiver,:],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],step=0.2)#trim=0.00000000001 is no trimming, the higher the more trimming
# Plots 'source'
plt.plot(earthquake_xyz[ind_receiver,0],earthquake_xyz[ind_receiver,1],earthquake_xyz[ind_receiver,2],'*r',markersize=10)
# Plots path
plt.plot(paths1[0,:],paths1[1,:],paths1[2,:],'g--',markersize=20)
# Plots endpoint
plt.plot(receiver_xyz[ind_receiver,0],receiver_xyz[ind_receiver,1],receiver_xyz[ind_receiver,2],'vb',markersize=10)
plt.title('Earthquake as the source')
plt.savefig(fname='earthquake_source.png',format='png',dpi=300)
plt.show()


## Plots raypath considering the RECEIVER as the source
t,d,a=fmm.eikonal(vp_velocity_3d.flatten(order='F'),xyz=receiver_xyz[ind_receiver,:],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2,angle=True)
time=t.reshape(nx,ny,nz,order='F')
time_zxy=np.transpose(time,(2,0,1))

plot3d(time_zxy,x=x_km,y=y_km,z=z_km,frames=[0,nx-1,0],cmap=plt.cm.jet,barlabel='Traveltime (s)',showf=False,close=False,alpha=0.7)
paths=ray3d(time,source=receiver_xyz[ind_receiver,:],receiver=earthquake_xyz[ind_receiver,:3],ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],step=0.2)#trim=0.00000000001 is no trimming, the higher the more trimming
# Plots 'source'
plt.plot(receiver_xyz[ind_receiver,0],receiver_xyz[ind_receiver,1],receiver_xyz[ind_receiver,2],'*r',markersize=10)
# Plots path
h1=plt.plot(paths[0,:],paths[1,:],paths[2,:],'k',markersize=20)
h2=plt.plot(paths1[0,:],paths1[1,:],paths1[2,:],'g--',markersize=20)
# Plots endpoint
plt.plot(earthquake_xyz[ind_receiver,0],earthquake_xyz[ind_receiver,1],earthquake_xyz[ind_receiver,2],'vb',markersize=10)
plt.title('Receiver as the source V.S. Earthquake as the source')
plt.legend([h1[0],h2[0]],['Ray 1 ( Receiver to Earthquake)', 'Ray 2 (Earthquake to Receiver)'], loc='lower left')
plt.savefig(fname='test_pyekfmm_raytracing3d_reciprocal.png',format='png',dpi=300)
plt.show()


