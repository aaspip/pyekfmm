from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import os


import pyekfmm as fmm
import numpy as np

##Event lat/lon/dep
evlat=-25+90
evlon=0+180
evdep=0

## size of rtp axes (number of samples)
numr=1
numt=1000
nump=2000

## beg/end in r
begr=6400
endr=6400

begt=0
endt=180

## beg/end in p
begp=0
endp=360

## sampling in r
dr=(endr-begr)/numr
dt=(endt-begt)/numt
dp=(endp-begp)/nump


vel=3.0*np.ones([numr*numt*nump,1],dtype='float32');
t=fmm.eikonal_rtp(vel,rtp=np.array([6400-evdep,evlat,evlon]),ar=[begr,dr,numr],at=[begt,dt,numt],ap=[begp,dp,nump],order=1);#spherical needs order=1
time=t.reshape(numr,numt,nump,order='F'); #[r,t,p]

vel=3.0*np.ones([numr*numt*nump,1],dtype='float32');
t=fmm.eikonal_rtp(vel,rtp=np.array([6400-evdep,180,evlon]),ar=[begr,dr,numr],at=[begt,dt,numt],ap=[begp,dp,nump],order=1);#spherical needs order=1
time2=t.reshape(numr,numt,nump,order='F'); #[r,t,p]

# import matplotlib.pyplot as plt
# # plt.figure(figsize=(12, 7))
# plt.imshow(time[numr-1,:,:],extent=[begp-180,endp-180,endt-90,begt-90],aspect='auto')#,clim=(0, 2000.0)); #rtp->tp
# plt.gca().invert_yaxis()
# plt.xlabel('Lon (deg)'); 
# plt.ylabel('Lat (deg)');
# plt.jet()
# plt.colorbar(orientation='horizontal',shrink=0.6,label='Traveltime (s)');
# plt.plot(evlon-180,evlat-90,'*',color='r',markersize=15)
# # plt.gca().invert_yaxis()
# plt.savefig('test_14_global.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
# plt.savefig('test_14_global.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)
# plt.show()

## Verify
print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
print(['Correct result:',6698.3594, 0.7446289, 1347.9429, 1816950.0])
#print(['Correct result:',285.1792, 0.016649984, 75.70713, 5731.57]) #If using order=2




# import os
# # read in topo data (on a regular lat/lon grid)
# etopo=np.loadtxt(os.environ["HOME"]+'/chenyk.data/cyk_small_dataset/etopo20data.gz')
# lons=np.loadtxt(os.environ["HOME"]+'/chenyk.data/cyk_small_dataset/etopo20lons.gz')
# lats=np.loadtxt(os.environ["HOME"]+'/chenyk.data/cyk_small_dataset/etopo20lats.gz')

lats=np.linspace((begt-90),(endt-90),numt);
lons=np.linspace((begp-180),(endp-180),nump);

# fig=plt.figure(figsize=(8, 8));
fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1.5]})
# plt.subplot(2,1,1);

plt.subplot(2,1,1);
m = Basemap(projection='moll',lon_0=0)
# make filled contour plot.
x, y = m(*np.meshgrid(lons, lats)) #strange here (1081,540)->meshgrid-> [540,1081)
cs = m.contourf(x,y,np.squeeze(time),30,cmap=plt.cm.jet)
# draw coastlines.
m.drawcoastlines()
# draw parallels and meridians.
m.drawparallels(np.arange(-60.,90.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(0.,360.,60.),labels=[0,0,0,1],fontsize=12)
m.colorbar(location='bottom',pad='10%',label='Traveltime (s)')
x_s,y_s=m(evlon-180,evlat-90)
plt.plot(x_s,y_s,'*',color='r',markersize=10)

cb = plt.colorbar(cs, cax = fig.add_axes([0.37,0.1,0.3,0.02]), format= "%4.0f", orientation='horizontal',label='Traveltime (s)')

cax = fig.add_axes([0.15,0.9,0.2,0.2])
plt.text(0,0, "a)", fontsize=28, color='k')
plt.axis('off')
# plt.show()

plt.subplot(2,1,2);
plt.axis('off')

## 
##Event lat/lon/dep
evlat=-25+90
evlon=0+180
evdep=400

## size of rtp axes (number of samples)
numr=2000
numt=1
nump=2000

## beg/end in r
begr=3200
endr=6400

begt=evlat
endt=evlat

## beg/end in p
begp=90
endp=270

## sampling in r
dr=(endr-begr)/numr
dt=(endt-begt)/numt
dp=(endp-begp)/nump


vel=6.0*np.ones([numr*numt*nump,1],dtype='float32');
t=fmm.eikonal_rtp(vel,rtp=np.array([6400-evdep,evlat,evlon]),ar=[begr,dr,numr],at=[begt,dt,numt],ap=[begp,dp,nump],order=1);#spherical needs order=1
time=t.reshape(numr,nump,order='F'); #[r,t,p]

vel=6.0*np.ones([numr*numt*nump,1],dtype='float32');
t=fmm.eikonal_rtp(vel,rtp=np.array([6400-evdep,180,evlon]),ar=[begr,dr,numr],at=[begt,dt,numt],ap=[begp,dp,nump],order=1);#spherical needs order=1
time2=t.reshape(numr,nump,order='F'); #[r,t,p]


## plot on polar coordinates
used_rad=np.linspace(begr,endr,numr)
used_theta=np.linspace((begp-180)/180*np.pi,(endp-180)/180*np.pi,nump)
theta,rad = np.meshgrid(used_theta, used_rad) #rectangular plot of polar data
X = theta
Y = rad

# plt.subplot(2,2,3);
plt.jet()
ax = fig.add_subplot(212,projection='polar')
cm=ax.pcolormesh(X, Y, time) #X,Y & data2D must all be same dimensions

ax.plot(0, 6400-evdep,'*',color='r',markersize=15)

ax.set_thetamin(begp)
ax.set_thetamax(endp)
ax.set_thetamin(-90)
ax.set_thetamax(+90)
ax.set_rorigin(5800-5700)
ax.set_theta_zero_location('W', offset=90)
ax.set_theta_zero_location('N', offset=0)
# ax.set_rmax(2)
ax.set_rticks([3200])  # less radial ticks
ax.set_rlabel_position([-22.5])  # get radial labels away from plotted line
ax.grid(True)
# ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Longitude (deg)',rotation=70)
# ---- mod here ---- #
# ax.set_theta_zero_location("N")  # theta=0 at the top
ax.set_theta_direction(-1)  # theta increasing clockwise
# plt.xticks([-45,0,45],['45$^o$S','0$^o$','45$^o$N'])
# plt.yticks([3200],['90'])
# 
cb = plt.colorbar(cm, cax = fig.add_axes([0.37,0.1,0.3,0.02]), format= "%4.0f", orientation='horizontal',label='Traveltime (s)')

# ax.text(-2000, 20, 'Longitude (deg)',  rotation=0)
ax.text(-2000, 1500, 'Depth (km)', rotation=0)

cax = fig.add_axes([0.15,0.4,0.2,0.2])
plt.text(0,0, "b)", fontsize=28, color='k')
plt.axis('off')

# # plt.yticks([0,-60],['00','60 S']) #not work
# # plt.yticks([3200],['90']) #not work
# plt.savefig('test_pyekfmm_fig5.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig5.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
# 
# # add a title.
plt.show()
