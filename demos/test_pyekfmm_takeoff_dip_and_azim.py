# A DEMO for computing time,dip,azim (traveltime and takeoff angle)
# Yangkang Chen, Feb 2, 2025
# 
import numpy as np
import pyekfmm as fmm
from pyseistr import plot3d
import matplotlib.pyplot as plt

'''
Computes the travel time, dip, and azimuth for a simple model with (possibly) analytical answer
'''

vp_velocity_3d=3.0*np.ones([100,100,100]);nx=100;ny=100;nz=100;dx=0.1;dy=0.1;dz=0.1;
## option 1
## Computes the takeoffs, dips, and azimuths for all earthquakes for the domain
t,d,a=fmm.eikonal(vp_velocity_3d.flatten(order='F'),xyz=np.array([int(nx/2),int(ny/2),int(nz/2)]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2,angle=True);


# plot noisy patches
plt.figure(figsize=(18,18))
ax=plt.subplot(2,2,1,projection='3d')
# plot3d(Xnoisy[600+ii,:].reshape(16,10,10,order='F'),ifnewfig=False,showf=False,close=False)

plot3d(np.transpose(vp_velocity_3d.reshape(nx,ny,nz,order='F'),(2,0,1)),frames=[0,99,0],vmin=1,vmax=1,cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Velocity',ifnewfig=False,showf=False,close=False,levels=[3.0000,3.000000001],alpha=0.75)
plt.plot(int(nx/2)*dx,int(ny/2)*dy,int(nz/2)*dy,'r*',markersize=20);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.title('Veloicty = 3.0 km/s')
# plt.colorbar(orientation='horizontal',cax=plt.gcf().add_axes([0.05,0.05,0.08,0.015]),shrink=1, format="%4.2e", ticks=[0,1.0]);

ax=plt.subplot(2,2,2,projection='3d')
# plot3d(Xnoisy[600+ii,:].reshape(16,10,10,order='F'),ifnewfig=False,showf=False,close=False)

plot3d(np.transpose(t.reshape(nx,ny,nz,order='F'),(2,0,1)),frames=[0,99,0],cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Time (s)',ifnewfig=False,showf=False,close=False)
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.title('Traveltime')

ax=plt.subplot(2,2,3,projection='3d')
plot3d(np.transpose(d.reshape(nx,ny,nz,order='F'),(2,0,1)),frames=[0,99,0],figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Dip (deg)',ifnewfig=False,showf=False,close=False)
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.title('Dip')

ax=plt.subplot(2,2,4,projection='3d')
plot3d(np.transpose(a.reshape(nx,ny,nz,order='F'),(2,0,1)),frames=[0,99,0],figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Azimuth (deg)',ifnewfig=False,showf=False,close=False)
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.title('Azimuth')

plt.savefig(fname='test_pyekfmm_takeoff_dip_and_azim.png',format='png',dpi=300)
# plt.gca().axis('off')
# ax = plt.gcf().add_subplot(111, aspect='auto');plt.gca().axis('off');plt.title("Noisy Patches",size=20)
plt.show()


