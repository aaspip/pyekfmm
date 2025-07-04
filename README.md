**Pyekfmm**
======

## Description

**Pyekfmm** is python package for 3D fast-marching-based traveltime calculation and its applications in seismology. The initial version of this package was held at https://github.com/chenyk1990/pyekfmm, which is no longer maintained.


## Reference
    Chen Y., Chen, Y.F., Fomel, S., Savvaidis, A., Saad, O.M., Oboue, Y.A.S.I. (2023). A python package for 3D fast-marching-based traveltime calculation and its applications in seismology, Seismological Research Letters, 94, 2050-2059.
    
BibTeX:

	@article{pyekfmm,
	  title={Pyekfmm: a python package for 3D fast-marching-based traveltime calculation and its applications in seismology},
	  author={Yangkang Chen and Yunfeng Chen and Sergey Fomel and Alexandros Savvaidis and Omar M. Saad and Yapo Abol\'{e} Serge Innocent Obou\'{e}},
	  journal={Seismological Research Letters},
	  volume={94},
	  number={1},
	  issue={1},
	  pages={2050-2059},
	  year={2023}
 	}

-----------
## Copyright
    pyekfmm developing team, 2021-present

-----------
## License
    MIT License 

-----------
## Install
Using the latest version

    git clone https://github.com/aaspip/pyekfmm
    cd pyekfmm
    pip install -v -e .

or using Pypi

    pip install pyekfmm

or (recommended, because we update very fast)

	pip install git+https://github.com/aaspip/pyekfmm


-----------
## Verified runnable OS
Mac OS, Linux, Windows (need Microsoft C++ Build Tools)


-----------
## Examples
    The "demo" directory contains all runable scripts to demonstrate different applications of pyekfmm. 

-----------
## Notebook tutorials
Some notebook tutorials are stored separately to ensure the minimal size of the pyekfmm package. They can be found at 

	https://github.com/aaspip/notebook/blob/main/pyekfmm

-----------
## Dependence Packages
* scipy 
* numpy 
* matplotlib

-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## Gallery
The gallery figures of the pyekfmm package can be found at
    https://github.com/aaspip/gallery/tree/main/pyekfmm
Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory with the exactly the same file name. These gallery figures are also presented below. 

DEMO1 
The following figure shows an example of traveltime calculation for 2D isotropic media (a) and anisotropic media (b). Generated by [demos/test_pyekfmm_fig1.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig1.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig1.png' alt='DEMO1' width=960/>

DEMO2
The following figure shows an example of traveltime calculation for 3D isotropic media (a) and anisotropic media (b). Generated by [demos/test_pyekfmm_fig2.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig2.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig2.png' alt='DEMO2' width=960/>

DEMO3 
The following figure shows an example of traveltime calculation for 2D heterogeneous isotropic and anisotropic media. (a) Vertical velocity model. (b) Horizontal velocity model. (c) Anisotropic parameter η model. (d) Traveltime table in isotropic media. (e) Traveltime table in anisotropic media. Generated by [demos/test_pyekfmm_fig3.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig3.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig3.png' alt='DEMO3' width=960/>

DEMO4 
The following figure shows an ray tracing example in 2D (a) and 3D (b) media with vertically increasing velocities. Generated by [demos/test_pyekfmm_fig4.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig4.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig4.png' alt='DEMO4' width=960/>

DEMO5 
The following figure shows an example of traveltime calculation for the global earth. Generated by [demos/test_pyekfmm_fig5.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig5.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig5.png' alt='DEMO5' width=960/>

DEMO6
The following figure shows a location example and comparison with the NonLinLoc (NLL) result. 
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig6.png' alt='DEMO6' width=960/>

DEMO7
The following figure shows a relocation example of the Spanish Springs, Nevada earthquake sequence. 
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig7.png' alt='DEMO7' width=960/>

DEMO8
The following figure shows a surface-wave tomography test. (a) Ray path between a pair of virtual source (red star) and station (blue triangle). The background is the 5 sec group velocities of the Australian continent from ambient noise imaging. (b) Travel time field. (c) Ray paths of all 25,899 pairs. (d)-(f) The same as (a)-(c) but for the initial model with a constant velocity.
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig8.png' alt='DEMO8' width=960/>

DEMO9
The following figure shows the traveltime misfit in the surface-wave tomography test. (a) Group velocities inverted from the travel time residuals using the kernel constructed from the initial model. (b) Travel time misfits estimated from the initial and final models.
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig9.png' alt='DEMO9' width=960/>

# Below are new examples in addition to the results in the original paper
DEM10
The following figure shows an example of traveltime calculation of two shots for 3D isotropic media
 Generated by [demos/test_pyekfmm_fig2-multishots.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig2.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig2-multishots.png' alt='DEMO2' width=960/>

DEM11
The following figure shows an example of traveltime calculation comparison between Pyekfmm and skfmm (scikit-fmm)
 Generated by [demos/test_pyekfmm_fig1.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig1.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig1-comp.png' alt='DEMO2' width=960/>

The following figure shows an example of traveltime calculation comparison between Pyekfmm and pykonal (if pykonal is installed)
 Generated by [demos/test_pyekfmm_fig1.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_fig1.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_fig1-comp2.png' alt='DEMO2' width=960/>


DEM12
The following figure shows an example of computing traveltime, takeoff angle (dip and azimuth)
 Generated by [demos/test_pyekfmm_takeoff_dip_and_azim.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_takeoff_dip_and_azim.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_takeoff_dip_and_azim-middle.png' alt='DEMO2' width=960/>

The following figure shows a slightly changed example of computing traveltime, takeoff angle (dip and azimuth) (from a source at the corner)
 Generated by [demos/test_pyekfmm_takeoff_dip_and_azim.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_takeoff_dip_and_azim.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_takeoff_dip_and_azim-corner.png' alt='DEMO2' width=960/>


DEM13
The following figure shows an example of computing rays in models with different sizes
 Generated by [demos/test_pyekfmm_raytracing3d.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_raytracing3d.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_raytracing3d.png' alt='DEMO2' width=960/>

DEM14
The following figure shows an example of computing reciprocal rays 
 Generated by [demos/test_pyekfmm_raytracing3d_reciprocal.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_raytracing3d_reciprocal.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_raytracing3d_reciprocal.png ' alt='DEMO2' width=960/>

DEM15
The following figure shows an example of benchmarking Pyekfmm with bh_tomo package 
 Generated by [demos/test_pyekfmm_raytracing2d_benchmarkWITHbhtomo.py](https://github.com/aaspip/pyekfmm/blob/main/demos/test_pyekfmm_raytracing2d_benchmarkWITHbhtomo.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyekfmm/test_pyekfmm_raytracing2d_benchmarkWITHbhtomo.png ' alt='DEMO2' width=960/>

