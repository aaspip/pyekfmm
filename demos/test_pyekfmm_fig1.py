## This DEMO is a 2D example [x,z] with constant velocity in VTI media and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2022, The University of Texas at Austin

import pyekfmm as fmm
import numpy as np

# vel=3.80395*np.ones([501*501,1],dtype='float32'); #velocity axis must be x,y,z respectively
# velx=3.09354*np.ones([501*501,1],dtype='float32'); #velocity axis must be x,y,z respectively
# eta=0.340859*np.ones([501*501,1],dtype='float32'); #velocity axis must be x,y,z respectively
velz=3.09354*np.ones([201*201,1],dtype='float32'); #velocity axis must be x,y,z respectively
velx=3.80395*np.ones([201*201,1],dtype='float32'); #velocity axis must be x,y,z respectively
eta=0.340859*np.ones([201*201,1],dtype='float32'); #velocity axis must be x,y,z respectively
t=fmm.eikonalvti(velx,velz,eta,xyz=np.array([5,0,5]),ax=[0,0.05,201],ay=[0,0.05,1],az=[0,0.05,201],order=1);
time=t.reshape(201,201,order='F');#first axis (vertical) is x, second is z

## Isotropic case
t=fmm.eikonal(velz,xyz=np.array([5,0,5]),ax=[0,0.05,201],ay=[0,0.05,1],az=[0,0.05,201],order=2);
time0=t.reshape(201,201,order='F');#first axis (vertical) is x, second is z

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121,aspect=1.0)
# plt.imshow(time.transpose(),cmap=plt.cm.jet, interpolation='none', extent=[0,10,10,0]); #transpose so that first axis is z, second is x
plt.contour(time0.transpose(),np.arange(10)*0.2,extent=[0,10,10,0])
plt.plot(5,5,'*r',markersize=10)
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.gca().invert_yaxis()
plt.text(-1.5, -0.5, 'a)', fontsize=24)

ax = fig.add_subplot(122,aspect=1.0)
plt.contour(time.transpose(),np.arange(10)*0.2,extent=[0,10,10,0])
plt.plot(5,5,'*r',markersize=10)
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.gca().invert_yaxis()
plt.text(-1.5, -0.5, 'b)', fontsize=24)

plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.02]),shrink=1,label='Traveltime (s)');


plt.savefig('test_pyekfmm_fig1.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig1.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)
plt.show()

## Verify
print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
print(['Correct result:',2.1797864, 0.0, 0.44571817, 0.19866468])


##########################################################################################
### Below is comparison with scikit-fmm
##########################################################################################
# pip install scikit-fmm
import skfmm

nx=201;nz=201;
xs=5;zs=5;dx=0.05;dz=0.05;ox=0;oz=0;
phi = np.ones((nx, nz))
# phi[int(xs // dx), int(zs // dz)] = -1.
phi[100, 100] = -1.
time_skfmm = skfmm.travel_time(phi, velz.reshape(201,201,order='F'), dx=(dx, dz))

time_true=np.zeros([nx,nz]);
for ii in range(nz):
	for jj in range(nx):
		time_true[jj,ii] = np.sqrt((ox+jj*dx-xs)*(ox+jj*dx-xs)+(oz+ii*dz-zs)*(oz+ii*dz-zs))/3.09354;


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(221,aspect=1.0)
plt.imshow(time0.transpose(),clim=(0, 2),extent=[0,10,10,0])
plt.plot(5,5,'*r',markersize=10)
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.gca().invert_yaxis()
plt.gca().text(-0.15,1,'a)',transform=plt.gca().transAxes,size=20,weight='normal')
plt.title('Traveltime (Pyekfmm)');

ax = fig.add_subplot(222,aspect=1.0)
plt.imshow(time_skfmm.transpose(),clim=(0, 2),extent=[0,10,10,0])
plt.plot(5,5,'*r',markersize=10)
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.gca().invert_yaxis()
plt.gca().text(-0.15,1,'b)',transform=plt.gca().transAxes,size=20,weight='normal')
plt.title('Traveltime (skfmm)');

ax = fig.add_subplot(223,aspect=1.0)
plt.imshow(np.abs(time_true.transpose()-time0.transpose())*100,clim=(0, 2),extent=[0,10,10,0])
plt.plot(5,5,'*r',markersize=10)
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.gca().invert_yaxis()
plt.gca().text(-0.15,1,'c)',transform=plt.gca().transAxes,size=20,weight='normal')
plt.title('Error*100 (Pyekfmm, MAE=%g)'%np.abs(time_true.transpose()-time0.transpose()).mean());

ax = fig.add_subplot(224,aspect=1.0)
plt.imshow(np.abs(time_true.transpose()-time_skfmm.transpose())*100,clim=(0, 2),extent=[0,10,10,0])

plt.plot(5,5,'*r',markersize=10)
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.gca().invert_yaxis()
plt.gca().text(-0.15,1,'d)',transform=plt.gca().transAxes,size=20,weight='normal')
plt.title('Error*100 (skfmm, MAE=%g)'%np.abs(time_true.transpose()-time_skfmm.transpose()).mean());

plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.02]),shrink=1,label='Traveltime (s)');
plt.savefig('test_pyekfmm_fig1-comp.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.show()


#order=1: Pyekfmm: MAE=0.0129547
#order=2: Pyekfmm: MAE=0.00265566
#skfmm: MAE=0.00540868



