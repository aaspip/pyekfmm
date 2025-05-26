## This DEMO is a 2D example [x,z] with constant velocity in VTI media and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2022, The University of Texas at Austin

import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

nz=240;
nx=737;
dx=12.5;dz=12.5;
xm=0+dx*(nx-1);zm=0+dz*(nz-1);
## Datapath: 
#https://github.com/aaspip/data/blob/main/marmvz.bin
#https://github.com/aaspip/data/blob/main/marmvx.bin
#https://github.com/aaspip/data/blob/main/marmeta.bin

#  VZ
fd = open('../data/marmousiVTI/marmvz.bin','rb')
vz = np.fromfile(fd, dtype = np.float32).reshape([nz,nx],order='F')    #[zxy]
plt.imshow(vz,extent=[0,(nx-1)*dx,(nz-1)*dz,0]);
plt.xlabel('X (m)');plt.ylabel('Z (m)');
plt.jet();
plt.colorbar(orientation='horizontal',label="Velocity (m/s)");
plt.show()

# VX
fd = open('../data/marmousiVTI/marmvx.bin','rb')
vx = np.fromfile(fd, dtype = np.float32).reshape([nz,nx],order='F')    #[zxy]
plt.imshow(vx,extent=[0,(nx-1)*dx,(nz-1)*dz,0]);
plt.xlabel('X (m)');plt.ylabel('Z (m)');
plt.jet();
plt.colorbar(orientation='horizontal',label="Velocity (m/s)");
plt.show()

# eta
fd = open('../data/marmousiVTI/marmeta.bin','rb')
et = np.fromfile(fd, dtype = np.float32).reshape([nz,nx],order='F')    #[zxy]
plt.imshow(et,extent=[0,(nx-1)*dx,(nz-1)*dz,0]);
plt.xlabel('X (m)');plt.ylabel('Z (m)');
plt.jet();
plt.colorbar(orientation='horizontal',label="Anisotropic parameter "+r'$\eta$');
plt.show()

# velx=3.80395*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
# eta=0.340859*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively

velx=vx.transpose().flatten(order='F') #[z,x] -> [x,z]
velz=vz.transpose().flatten(order='F')  #[z,x] -> [x,z]
eta=et.transpose().flatten(order='F') #[z,x] -> [x,z]

## should use order=1 for VTI
t=fmm.eikonalvti(velx,velz,eta,xyz=np.array([3000,0,0]),ax=[0,dx,nx],ay=[0,dx,1],az=[0,dz,nz],order=1);
time=t.reshape(nx,nz,order='F');#first axis (vertical) is x, second is z


## Isotropic case
t=fmm.eikonal(velz,xyz=np.array([3000,0,0]),ax=[0,dx,nx],ay=[0,dx,1],az=[0,dz,nz],order=1);
time0=t.reshape(nx,nz,order='F');#first axis (vertical) is x, second is z


fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(231,aspect=1.0)
plt.imshow(vz,extent=[0,(nx-1)*dx,(nz-1)*dz,0]);
plt.xlabel('X (m)');plt.ylabel('Z (m)');
plt.jet();
plt.colorbar(orientation='horizontal',aspect=25,label="Velocity (m/s)");
plt.text(-1300, -100, 'a)', fontsize=22)

ax = fig.add_subplot(232,aspect=1.0)
plt.imshow(vx,extent=[0,(nx-1)*dx,(nz-1)*dz,0]);
plt.xlabel('X (m)');#plt.ylabel('Y (m)');
plt.jet();
plt.colorbar(orientation='horizontal',aspect=25,label="Velocity (m/s)");
plt.text(-1300, -100, 'b)', fontsize=22)

ax = fig.add_subplot(233,aspect=1.0)
plt.imshow(et,extent=[0,(nx-1)*dx,(nz-1)*dz,0]);
plt.xlabel('X (m)');#plt.ylabel('Y (m)');
plt.jet();
plt.colorbar(orientation='horizontal',aspect=25,label="Anisotropic parameter "+r'$\eta$');
plt.text(-1300, -100, 'c)', fontsize=22)

ax = fig.add_subplot(223,aspect=1.0)
# plt.imshow(time.transpose(),cmap=plt.cm.jet, interpolation='none', extent=[0,10,10,0]); #transpose so that first axis is z, second is x
plt.contour(time0.transpose(),np.arange(13)*0.181,extent=[0,xm,0,zm]);
plt.gca().invert_yaxis()
plt.plot(3000,50,'*r',markersize=10)
plt.xlabel('X (m)');plt.ylabel('Z (m)');
plt.jet()
plt.text(-1300, -100, 'd)', fontsize=22)

ax = fig.add_subplot(224,aspect=1.0)
plt.contour(time.transpose(),np.arange(13)*0.181,extent=[0,xm,0,zm]);
plt.gca().invert_yaxis()
plt.plot(3000,50,'*r',markersize=10)
plt.xlabel('X (m)');plt.ylabel('Z (m)');
plt.jet()
plt.text(-1300, -100, 'e)', fontsize=22)

plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.02]),shrink=1,label='Traveltime (s)');
plt.savefig('test_pyekfmm_fig3.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig3.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)
plt.show()

## Verify
print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
print(['Correct result:',2.84689, 0.0, 0.53170025, 0.28270516])

