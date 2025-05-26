#This script reproduces figure 3 in the Pykonal paper; the original Pykonal script is in the Pykonal folder
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np


# ## Initialize a solver in Cartesian coordinates
# Using pykonal (if you install pykonal)
# import pykonal
# solver_c = pykonal.EikonalSolver(coord_sys='cartesian')
# solver_c.velocity.min_coords = -25, -25, 0
# solver_c.velocity.node_intervals = 1, 1, 1
# solver_c.velocity.npts = 51, 51, 1
# solver_c.velocity.values = np.ones(solver_c.velocity.npts) # Uniform velocity model
# 
# # Add a point source at the origin of the grid.
# src_idx = (25, 25, 0)
# solver_c.traveltime.values[src_idx] = 0
# solver_c.unknown[src_idx] = False
# solver_c.trial.push(*src_idx)
# 
# # Solve the system.
# solver_c.solve()
################################################################################################
# Using pyekfmm
## Isotropic case (Cartesian)
################################################################################################
import pyekfmm as fmm
nz,nx=51,51
x=-25+np.arange(nx)*1
z=-25+np.arange(nz)*1
xx_c,yy_c=np.meshgrid(x,z) #for plotting
xx_c=xx_c.transpose() #for plotting
yy_c=yy_c.transpose() #for plotting

vel=np.ones(nz*nx)
t=fmm.eikonal(vel,xyz=np.array([0,0,0]),ax=[-25,1,nx],ay=[0,0.05,1],az=[-25,1,nz],order=2);
time_c=t.reshape(nx,nz,order='F');	#first axis (vertical) is x, second is z
time_c=time_c.transpose()	   #NOW: first axis (vertical) is z, second is x


# ## Initialize a solver in Spherical coordinates
# Using pykonal (if you install pykonal)
# solver_s = pykonal.EikonalSolver(coord_sys='spherical')
# solver_s.velocity.min_coords = 0.01, np.pi/2, 0
# solver_s.velocity.node_intervals = 1, 1, np.pi/10
# solver_s.velocity.npts = 25, 1, 21
# solver_s.velocity.values = np.ones(solver_s.velocity.npts) # Uniform velocity model
# 
# # Add a point source at the origin of the grid.
# for it in range(solver_s.velocity.npts[1]):
#     for ip in range(solver_s.velocity.npts[2]):
#         src_idx = (0, it, ip)
#         vv = solver_s.velocity.values[src_idx]
#         solver_s.traveltime.values[src_idx] = solver_s.velocity.nodes[src_idx + (0,)] / vv
#         solver_s.unknown[src_idx] = False
#         solver_s.trial.push(*src_idx)
#         
# # Solve the system
# # get_ipython().run_line_magic('time', 'solver_s.solve()')
# solver_s.solve()

################################################################################################
## Isotropic case (Spherical)
################################################################################################
nr,nt,nph=25,1,21 #nph->nphi, no np as a variable in python, #NOTE: change 21->51, the error in (d) will disappear
vel=np.ones(nr*nt*nph,dtype='float32')
r0=0;dr=1;
t0=np.pi/2/(np.pi)*180;dt=0;#when t0=90 or pi/2, sin(t0)=1
p0=0;dp=np.pi/10/(np.pi)*180

r=r0+np.arange(nr)*dr
p=p0+np.arange(nph)*dp

t=fmm.eikonal_rtp(vel,rtp=np.array([0,t0,0]),ar=[r0,dr,nr],at=[t0,dt,nt],ap=[p0,dp,nph],order=1);#spherical needs order=1
time_s=t.reshape(nr,nph,order='F'); #[r,t,p]

# print('Norm(time_s-solver_s.traveltime.values)=',np.linalg.norm(time_s-solver_s.traveltime.values[:,0,:]))
# 
# plt.imshow(np.concatenate([solver_s.traveltime.values[:,0,:],time_s,(solver_s.traveltime.values[:,0,:]-time_s)*100],axis=1))
# plt.colorbar();
# plt.savefig('Spherical.png')
# plt.show()

iz, it = 0, 0
plot_kwargs = dict(
    shading='gouraud'
)
plt.close('all')

# nodes = solver_s.traveltime.nodes
# xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
# yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
rr,pp=np.meshgrid(r,p)
rr=rr.transpose()
pp=pp.transpose()

xx=rr*np.sin(t0/180*np.pi)*np.cos(pp/180*np.pi)
yy=rr*np.sin(t0/180*np.pi)*np.sin(pp/180*np.pi)

fig = plt.figure(figsize=(5, 4))

gridspec = gs.GridSpec(nrows=2, ncols=4, width_ratios=[0.08, 1, 1, 0.08])
ax1 = fig.add_subplot(gridspec[0, 1], aspect=1)
ax2 = fig.add_subplot(gridspec[0, 2], aspect=1)
ax3 = fig.add_subplot(gridspec[1, 1], aspect=1)
ax4 = fig.add_subplot(gridspec[1, 2], aspect=1)
cax1 = fig.add_subplot(gridspec[0, 3])
cax2 = fig.add_subplot(gridspec[1, 3])
ax1.set_title('Cartesian')
ax2.set_title('Spherical')
qmesh = ax1.pcolormesh(
    x,
    z,
    time_c,
    cmap=plt.get_cmap('jet_r'),
    **plot_kwargs
)

qmesh = ax2.pcolormesh(
    xx,
    yy,
    time_s,
    vmin=qmesh.get_array().min(),
    vmax=qmesh.get_array().max(),
    cmap=plt.get_cmap('jet_r'),
    **plot_kwargs
)
cbar1 =  fig.colorbar(qmesh, cax=cax1)
cbar1.set_label('Travel time [s]')

qmesh = ax3.pcolormesh(
    xx_c,
    yy_c,
    time_c-np.sqrt(xx_c**2 + yy_c**2),
    vmin=0.0,
    vmax=0.35,
    cmap=plt.get_cmap('bone_r'),
    **plot_kwargs
)

qmesh = ax4.pcolormesh(
    xx,
    yy,
    time_s-np.sqrt(xx**2 + yy**2),
    vmin=0.0,
    vmax=0.35,
    cmap=plt.get_cmap('bone_r'),
    **plot_kwargs
)

cbar2 =  fig.colorbar(qmesh, cax=cax2)
cbar2.set_label('Absolute error [s]')

for ax in (ax1, ax2):
    ax.set_xticklabels([])
    
for ax in (ax2, ax4):
    ax.set_yticklabels([])
    
panel = ord("a")
for ax in (ax1, ax2, ax3, ax4):
    ax.text(
        0, 1.025, f"({chr(panel)})",
        ha="right",
        va="bottom",
        transform=ax.transAxes
    )
    panel += 1

fig.text(0.5, 0, "x [km]", ha="center", va="bottom")
fig.text(0.05, 0.5, "y [km]", ha="left", va="center", rotation=90)

plt.savefig('test_pykonal_figure_3.png',bbox_inches='tight')
plt.show()






