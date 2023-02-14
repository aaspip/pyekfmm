#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#define FMM_HUGE 9999999999999999

/*****pqueue for neighbor***/
enum {FMM_IN, FMM_FRONT, FMM_OUT};

static float **x, **xn, **x1;

void pqueue_init (int n)
/*< Initialize heap with the maximum size >*/
{
    x = (float **) malloc ((n+1)*sizeof (float *));
}

void pqueue_start (void)
/*< Set starting values >*/
{
    xn = x;
    x1 = x+1;
}

void pqueue_close (void)
/*< Free the allocated storage >*/
{
    free (x);
}

float* pqueue_extract (void)
/*< Extract the smallest element >*/
{
    unsigned int c;
    int n;
    float *v, *t;
    float **xi, **xc;
    
    v = *(x1);
    *(xi = x1) = t = *(xn--);
    n = (int) (xn-x);
    if (n < 0) return NULL;
    for (c = 2; c <= (unsigned int) n; c <<= 1) {
	xc = x + c;
	if (c < (unsigned int) n && **xc > **(xc+1)) {
	    c++; xc++;
	}
	if (*t <= **xc) break;
	*xi = *xc; xi = xc;
    }
    *xi = t;
    return v;
}

/**** pqueue **/


void pqueue_insert (float* v)
/*< Insert an element (smallest first) >*/
{
    float **xi, **xq;
    unsigned int q;
    
    xi = ++xn;
    *xi = v;
    q = (unsigned int) (xn-x);
    for (q >>= 1; q > 0; q >>= 1) {
	xq = x + q;
	if (*v > **xq) break;
	*xi = *xq; xi = xq;
    }
    *xi = v; 
}
/****pqueue for neighbor***/


/***neighbor.c*/
struct Upd {
    double stencil, value;
    double delta;
};

static int update (float value, int i);
static float qsolve(int i); 
static float qsolve_rtp(int i); 
static void stencil (float t, struct Upd *x); 
static bool updaten (int m, float* res, struct Upd *v[]);
static void grid (int *i, const int *n);

static int *in, *n, s[3], order;
static float *ttime, *vv, rdx[3];
static float dd[3], oo[3]; /*sampling intervals in r,t,p*/
static double v1;

void neighbors_init (int *in1     /* status flag [n[0]*n[1]*n[2]] */, 
			float *rdx1  /* grid sampling [3] */, 
			int *n1      /* grid samples [3] */, 
			int order1   /* accuracy order */, 
			float *time1 /* traveltime [n[0]*n[1]*n[2]] */)
/*< Initialize >*/
{
    in = in1; ttime = time1; 
    n = n1; order = order1;
    s[0] = 1; s[1] = n[0]; s[2] = n[0]*n[1];
    rdx[0] = 1./(rdx1[0]*rdx1[0]);
    rdx[1] = 1./(rdx1[1]*rdx1[1]);
    rdx[2] = 1./(rdx1[2]*rdx1[2]);
}

int  neighbours(int i) 
/*< Update neighbors of gridpoint i, return number of updated points >*/
{
    int j, k, ix, npoints;
    
    npoints = 0;
    for (j=0; j < 3; j++) {
	ix = (i/s[j])%n[j];
	if (ix+1 <= n[j]-1) {
	    k = i+s[j]; 
	    if (in[k] != FMM_IN) npoints += update(qsolve(k),k);
	}
	if (ix-1 >= 0  ) {
	    k = i-s[j];
	    if (in[k] != FMM_IN) npoints += update(qsolve(k),k);
	}
    }
    return npoints;
}

int  neighbours_rtp(int i) 
/*< Update neighbors of gridpoint i, return number of updated points >*/
{
    int j, k, ix, npoints;
    
    npoints = 0;
    for (j=0; j < 3; j++) {
	ix = (i/s[j])%n[j];
	if (ix+1 <= n[j]-1) {
	    k = i+s[j]; 
	    if (in[k] != FMM_IN) npoints += update(qsolve_rtp(k),k);
	}
	if (ix-1 >= 0  ) {
	    k = i-s[j];
	    if (in[k] != FMM_IN) npoints += update(qsolve_rtp(k),k);
	}
    }
    return npoints;
}

static int update (float value, int i)
/* update gridpoint i with new value */
{
    if (value < ttime[i]) {
	ttime[i]   = value;
	if (in[i] == FMM_OUT) { 
	    in[i] = FMM_FRONT;      
	    pqueue_insert (ttime+i);
	    return 1;
	}
    }
    
    return 0;
}

static float qsolve(int i)
/* find new traveltime at gridpoint i */
{
    int j, k, ix;
    float a, b, t, res;
    struct Upd *v[3], x[3], *xj;

    for (j=0; j<3; j++) {
	ix = (i/s[j])%n[j];
	
	if (ix > 0) { 
	    k = i-s[j];
	    a = ttime[k];
	} else {
	    a = FMM_HUGE;
	}

	if (ix < n[j]-1) {
	    k = i+s[j];
	    b = ttime[k];
	} else {
	    b = FMM_HUGE;
	}

	xj = x+j;
	xj->delta = rdx[j];
	
	if (a < b) {
	    xj->stencil = xj->value = a;
	} else {
	    xj->stencil = xj->value = b;
	}

	if (order > 1) {
	    if (a < b  && ix-2 >= 0) { 
		k = i-2*s[j];
		if (in[k] != FMM_OUT && a >= (t=ttime[k]))
		    stencil(t,xj);
	    }
	    if (a > b && ix+2 <= n[j]-1) { 
		k = i+2*s[j];
		if (in[k] != FMM_OUT && b >= (t=ttime[k]))
		    stencil(t,xj);
	    }
	}
    }

    if (x[0].value <= x[1].value) {
	if (x[1].value <= x[2].value) {
	    v[0] = x; v[1] = x+1; v[2] = x+2;
	} else if (x[2].value <= x[0].value) {
	    v[0] = x+2; v[1] = x; v[2] = x+1;
	} else {
	    v[0] = x; v[1] = x+2; v[2] = x+1;
	}
    } else {
	if (x[0].value <= x[2].value) {
	    v[0] = x+1; v[1] = x; v[2] = x+2;
	} else if (x[2].value <= x[1].value) {
	    v[0] = x+2; v[1] = x+1; v[2] = x;
	} else {
	    v[0] = x+1; v[1] = x+2; v[2] = x;
	}
    }
    
    v1=vv[i];

    if(v[2]->value < FMM_HUGE) {   /* ALL THREE DIRECTIONS CONTRIBUTE */
	if (updaten(3, &res, v) || 
	    updaten(2, &res, v) || 
	    updaten(1, &res, v)) return res;

    } else if(v[1]->value < FMM_HUGE) { /* TWO DIRECTIONS CONTRIBUTE */
	if (updaten(2, &res, v) || 
	    updaten(1, &res, v)) return res;

    } else if(v[0]->value < FMM_HUGE) { /* ONE DIRECTION CONTRIBUTES */
	if (updaten(1, &res, v)) return res;

    }
	
    return FMM_HUGE;
}

static float qsolve_rtp(int i)
/* find new traveltime at gridpoint i */
{
    int j, k, ix, id0, id1, id2;
    float a, b, t, res;
    struct Upd *v[3], x[3], *xj;

	id0=(i/s[0])%n[0]; /*r index*/
	id1=(i/s[1])%n[1]; /*t index*/
	id2=(i/s[2])%n[2]; /*p index*/

    for (j=0; j<3; j++) {
	ix = (i/s[j])%n[j];
	
	if (ix > 0) { 
	    k = i-s[j];
	    a = ttime[k];
	} else {
	    a = FMM_HUGE;
	}

	if (ix < n[j]-1) {
	    k = i+s[j];
	    b = ttime[k];
	} else {
	    b = FMM_HUGE;
	}

	xj = x+j;
	
	if(j==0)
	{
	xj->delta = rdx[j];
	}
	if(j==1)
	{
	/*Remember: r=0 is the singularity */
	xj->delta = rdx[j]/(oo[0]+id0*dd[0]+0.00000000001)/(oo[0]+id0*dd[0]+0.00000000001); /*1/r/r*/
// 	xj->delta = rdx[j]/(oo[0]+id0*dd[0])/(oo[0]+id0*dd[0]); /*1/r/r*/

	}
	if(j==2)
	{
	/*Remember: r=0 and t=0 are the singularities */
	xj->delta = rdx[j]/(oo[0]+id0*dd[0]+0.00000000001)/(oo[0]+id0*dd[0]+0.00000000001)/(sinf(oo[1]+id1*dd[1])+0.00000000001)/(sinf(oo[1]+id1*dd[1])+0.00000000001); /*1/r/r/sin[theta]/sin[theta]*/
	
// 		xj->delta = rdx[j]/(oo[0]+id0*dd[0])/(oo[0]+id0*dd[0])/(sinf(oo[1]+id1*dd[1]))/(sinf(oo[1]+id1*dd[1])); /*1/r/r/sin[theta]/sin[theta]*/
	}

	if (a < b) {
	    xj->stencil = xj->value = a;
	} else {
	    xj->stencil = xj->value = b;
	}

	if (order > 1) {
	    if (a < b  && ix-2 >= 0) { 
		k = i-2*s[j];
		if (in[k] != FMM_OUT && a >= (t=ttime[k]))
		    stencil(t,xj);
	    }
	    if (a > b && ix+2 <= n[j]-1) { 
		k = i+2*s[j];
		if (in[k] != FMM_OUT && b >= (t=ttime[k]))
		    stencil(t,xj);
	    }
	}
    }

    if (x[0].value <= x[1].value) {
	if (x[1].value <= x[2].value) {
	    v[0] = x; v[1] = x+1; v[2] = x+2;
	} else if (x[2].value <= x[0].value) {
	    v[0] = x+2; v[1] = x; v[2] = x+1;
	} else {
	    v[0] = x; v[1] = x+2; v[2] = x+1;
	}
    } else {
	if (x[0].value <= x[2].value) {
	    v[0] = x+1; v[1] = x; v[2] = x+2;
	} else if (x[2].value <= x[1].value) {
	    v[0] = x+2; v[1] = x+1; v[2] = x;
	} else {
	    v[0] = x+1; v[1] = x+2; v[2] = x;
	}
    }
    
    v1=vv[i];

    if(v[2]->value < FMM_HUGE) {   /* ALL THREE DIRECTIONS CONTRIBUTE */
	if (updaten(3, &res, v) || 
	    updaten(2, &res, v) || 
	    updaten(1, &res, v)) return res;

    } else if(v[1]->value < FMM_HUGE) { /* TWO DIRECTIONS CONTRIBUTE */
	if (updaten(2, &res, v) || 
	    updaten(1, &res, v)) return res;

    } else if(v[0]->value < FMM_HUGE) { /* ONE DIRECTION CONTRIBUTES */
	if (updaten(1, &res, v)) return res;

    }
	
    return FMM_HUGE;
}

static void stencil (float t, struct Upd *x)
/* second-order stencil */
{
    x->delta *= 2.25;
    x->stencil = (4.0*x->value - t)/3.0;
}

static bool updaten (int m, float* res, struct Upd *v[]) 
/* updating */
{
    double a, b, c, discr, t;
    int j;

    a = b = c = 0.;

    for (j=0; j<m; j++) {
	a += v[j]->delta;
	b += v[j]->stencil*v[j]->delta;
	c += v[j]->stencil*v[j]->stencil*v[j]->delta;
    }
    b /= a;

    discr=b*b+(v1-c)/a;

    if (discr < 0.) return false;
    
    t = b + sqrt(discr);
    if (t <= v[m-1]->value) return false;

    *res = t;
    return true;
}

static void grid (int *i, const int *n)
/* restrict i[3] to the grid n[3] */
{ 
    int j;

    for (j=0; j < 3; j++) {
	if (i[j] < 0) {
	    i[j]=0;
	} else if (i[j] >= n[j]) {
	    i[j]=n[j]-1;
	}
    }
}

int neighbors_nearsource(float* xs   /* source location [3] */, 
			    int* b      /* constant-velocity box around it [3] */, 
			    float* d    /* grid sampling [3] */, 
			    float* vv1  /* slowness [n[0]*n[1]*n[2]] */, 
			    bool *plane /* if plane-wave source */)
/*< initialize the source >*/
{
    int npoints, ic, i, j, is, start[3], endx[3], ix, iy, iz;
    double delta[3], delta2;
    

    /* initialize everywhere */
    for (i=0; i < n[0]*n[1]*n[2]; i++) {
	in[i] = FMM_OUT;
	ttime[i] = FMM_HUGE;
    }

    vv = vv1;

    /* Find index of the source location and project it to the grid */
    for (j=0; j < 3; j++) {
	is = xs[j]/d[j]+0.5;
	start[j] = is-b[j]; 
	endx[j]  = is+b[j];
    } 
    
    grid(start, n);
    grid(endx, n);
    
    ic = (start[0]+endx[0])/2 + 
	n[0]*((start[1]+endx[1])/2 +
	      n[1]*(start[2]+endx[2])/2);
    
    v1 = vv[ic];

    /* loop in a small box around the source */
    npoints = n[0]*n[1]*n[2];
    for (ix=start[2]; ix <= endx[2]; ix++) {
	for (iy=start[1]; iy <= endx[1]; iy++) {
	    for (iz=start[0]; iz <= endx[0]; iz++) {
		npoints--;
		i = iz + n[0]*(iy + n[1]*ix);

		delta[0] = xs[0]-iz*d[0];
		delta[1] = xs[1]-iy*d[1];
		delta[2] = xs[2]-ix*d[2];

		delta2 = 0.;
		for (j=0; j < 3; j++) {
		    if (!plane[2-j]) delta2 += delta[j]*delta[j];
		}

		/* analytical formula (Euclid) */ 
		ttime[i] = sqrtf(v1*delta2);
		in[i] = FMM_IN;

		if ((n[0] > 1 && (iz == start[0] || iz == endx[0])) ||
		    (n[1] > 1 && (iy == start[1] || iy == endx[1])) ||
		    (n[2] > 1 && (ix == start[2] || ix == endx[2]))) {
		    pqueue_insert (ttime+i);
		}
	    }
	}
    }
    
    return npoints;
}


int neighbors_nearsource_rtp(float* xs   /* source location [3] */, 
			    int* b      /* constant-velocity box around it [3] */, 
			    float* d    /* grid sampling [3] */, 
			    float* vv1  /* slowness [n[0]*n[1]*n[2]] */, 
			    bool *plane /* if plane-wave source */)
/*< initialize the source >*/
{
    int npoints, ic, i, j, is, start[3], endx[3], ix, iy, iz;
    double delta[3], delta2;

	dd[0]=d[0];dd[1]=d[1];dd[2]=d[2];
	
    /* initialize everywhere */
    for (i=0; i < n[0]*n[1]*n[2]; i++) {
	in[i] = FMM_OUT;
	ttime[i] = FMM_HUGE;
    }

    vv = vv1;

    /* Find index of the source location and project it to the grid */
    for (j=0; j < 3; j++) {
	is = xs[j]/d[j]+0.5;
	start[j] = is-b[j]; 
	endx[j]  = is+b[j];
    } 
    
    grid(start, n);
    grid(endx, n);

    ic = (start[0]+endx[0])/2 + 
	n[0]*((start[1]+endx[1])/2 +
	      n[1]*(start[2]+endx[2])/2);
    
    v1 = vv[ic];

    /* loop in a small box around the source */
    npoints = n[0]*n[1]*n[2];
    for (ix=start[2]; ix <= endx[2]; ix++) {
	for (iy=start[1]; iy <= endx[1]; iy++) {
	    for (iz=start[0]; iz <= endx[0]; iz++) {
		npoints--;
		i = iz + n[0]*(iy + n[1]*ix);

		delta[0] = xs[0]-iz*d[0];
		delta[1] = (xs[1]-iy*d[1])*(oo[0]+iz*d[0]); 			/*r*/
		delta[2] = (xs[2]-ix*d[2])*((oo[0]+iz*d[0])*sinf(oo[1]+iy*d[1]));/*r*sin(t)*/
		delta2 = 0.;
		for (j=0; j < 3; j++) {
		    if (!plane[2-j]) delta2 += delta[j]*delta[j];
		}

		/* analytical formula (Euclid) */ 
		ttime[i] = sqrtf(v1*delta2);
		in[i] = FMM_IN;

		if ((n[0] > 1 && (iz == start[0] || iz == endx[0])) ||
		    (n[1] > 1 && (iy == start[1] || iy == endx[1])) ||
		    (n[2] > 1 && (ix == start[2] || ix == endx[2]))) {
		    pqueue_insert (ttime+i);
		}
	    }
	}
    }
    
    return npoints;
}
/***neighbor.c*/


void fastmarch_init (int n3,int n2,int n1) 
/*< Initialize data dimensions >*/
{
    int maxband;
    
    maxband = 0;
    if (n1 > 1) maxband += 2*n2*n3;
    if (n2 > 1) maxband += 2*n1*n3;
    if (n3 > 1) maxband += 2*n1*n2;

    pqueue_init (10*maxband);
}

void fastmarch (float* time                /* time */, 
		float* v                   /* slowness squared */, 
		int* in                    /* in/front/out flag */, 
		bool* plane                /* if plane source */,
		int   n3,  int n2,  int n1 /* dimensions */,
		float o3,float o2,float o1 /* origin */,
		float d3,float d2,float d1 /* sampling */,
		float s3,float s2,float s1 /* source */,
		int   b3,  int b2,  int b1 /* box around the source */,
		int order                  /* accuracy order (1,2,3) */)
/*< Run fast marching eikonal solver >*/
{
    float xs[3], d[3], *p;
    int n[3], b[3], npoints, i;
    
    n[0] = n1; xs[0] = s1-o1; b[0] = b1; d[0] = d1;
    n[1] = n2; xs[1] = s2-o2; b[1] = b2; d[1] = d2;
    n[2] = n3; xs[2] = s3-o3; b[2] = b3; d[2] = d3;

    pqueue_start();
    neighbors_init (in, d, n, order, time);

    for (npoints =  neighbors_nearsource (xs, b, d, v, plane);
	 npoints > 0;
	 npoints -= neighbours(i)) {
	/* Pick smallest value in the NarrowBand
	   mark as good, decrease points_left */

	p = pqueue_extract();

	if (p == NULL) {
	    break;
	}
	
	i = p - time;

	in[i] = FMM_IN;
    }
}

void fastmarch_rtp (float* time                /* time */, 
		float* v                   /* slowness squared */, 
		int* in                    /* in/front/out flag */, 
		bool* plane                /* if plane source */,
		int   n3,  int n2,  int n1 /* dimensions */,
		float o3,float o2,float o1 /* origin */,
		float d3,float d2,float d1 /* sampling */,
		float s3,float s2,float s1 /* source */,
		int   b3,  int b2,  int b1 /* box around the source */,
		int order                  /* accuracy order (1,2,3) */)
/*< Run fast marching eikonal solver >*/
{
    float xs[3], d[3], *p;
    int n[3], b[3], npoints, i;
    
    n[0] = n1; xs[0] = s1-o1; b[0] = b1; d[0] = d1;
    n[1] = n2; xs[1] = s2-o2; b[1] = b2; d[1] = d2;
    n[2] = n3; xs[2] = s3-o3; b[2] = b3; d[2] = d3;

	oo[0]=o1;oo[1]=o2;oo[2]=o3;
	
    pqueue_start();
    neighbors_init (in, d, n, order, time);

    for (npoints =  neighbors_nearsource_rtp (xs, b, d, v, plane);
	 npoints > 0;
	 npoints -= neighbours_rtp(i)) {
	/* Pick smallest value in the NarrowBand
	   mark as good, decrease points_left */

	p = pqueue_extract();

	if (p == NULL) {
	    break;
	}
	
	i = p - time;

	in[i] = FMM_IN;
    }
}

void fastmarch_close (void)
/*< Free allocated storage >*/
{
    pqueue_close();
}


static PyObject *eikonalc_oneshot(PyObject *self, PyObject *args){

    /*Below is the input part*/
    float f1,f2,f3,f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd;

	PyArg_ParseTuple(args, "Offfffffffiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v;
    float x, y, z;
    bool plane[3];
    
	x=f1;
	y=f2;
	z=f3;
	
	o1=f4;
	o2=f5;
	o3=f6;
	
	d1=f7;
	d2=f8;
	d3=f9;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	
	
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    /*
     * my code starts here
     */
    nd=PyArray_NDIM(arr1);
    npy_intp *sp=PyArray_SHAPE(arr1);

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 

    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;

    /* File with shot locations (n2=number of shots, n1=3) */

	nshot = 1;
	ndim = 3;
     
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));

	s[0][0]=x;
	s[0][1]=y;
	s[0][2]=z;
	
    n123 = n1*n2*n3;

	t = (float*)malloc(n123 * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));
	

    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d \n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }
    
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    nshot=1;
    for( is = 0; is < nshot; is++) {
	fastmarch(t,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);
    }
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*n3;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = t[i];
	
	return PyArray_Return(vecout);
	
}

static PyObject *eikonalc_oneshot_angle(PyObject *self, PyObject *args){

    /*Below is the input part*/
    float f1,f2,f3,f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd;

	PyArg_ParseTuple(args, "Offfffffffiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v, *dip, *azim;
    float x, y, z;
    bool plane[3];
    
	x=f1;
	y=f2;
	z=f3;
	
	o1=f4;
	o2=f5;
	o3=f6;
	
	d1=f7;
	d2=f8;
	d3=f9;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	
	
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    /*
     * my code starts here
     */
    nd=PyArray_NDIM(arr1);
    npy_intp *sp=PyArray_SHAPE(arr1);

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 

    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;

    /* File with shot locations (n2=number of shots, n1=3) */

	nshot = 1;
	ndim = 3;
     
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));

	s[0][0]=x;
	s[0][1]=y;
	s[0][2]=z;
	
    n123 = n1*n2*n3;

	t = (float*)malloc(n123 * sizeof(float));
	dip = (float*)malloc(n123 * sizeof(float));
	azim = (float*)malloc(n123 * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));
	

    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d \n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }
    
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    nshot=1;
    for( is = 0; is < nshot; is++) {
	fastmarch(t,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);
    }

    int i1,i2,i3,if2d,iflip;
    float grad1,grad2,grad3;
    float cPI,cRPD;
    
    cPI = 4. * atan(1.); /* PI */
    cRPD = cPI / 180.; /* radians per degree */
    
    /*initialization of takeoff angle*/
    for(i=0;i<n123*nshot;i++)
    {dip[i]=200;azim[i]=400;}
    
    if(n1==1 || n2==1)
    {
    if2d=1;	
    }else{
    if2d=0; 
    }
    
    if(if2d==0) /*3D version*/
    for(is=0;is<nshot;is++)
	for(i1=1;i1<n1-1;i1++) /*x*/
	for(i2=1;i2<n2-1;i2++) /*y*/
	for(i3=1;i3<n3-1;i3++) /*z*/
    	{
    	/*reverse sign to get take-off angle*/
    	grad1=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1-1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d1 + 
    		  (t[i1+1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d1)/2.0;
    	
    	grad2=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+(i2-1)*n1+i3*n1*n2+is*n1*n2*n3])/d2 + 
    		  (t[i1+(i2+1)*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d2)/2.0;
    	
    	grad3=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+(i3-1)*n1*n2+is*n1*n2*n3])/d3 + 
    		  (t[i1+i2*n1+(i3+1)*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d3)/2.0;
    	
    	/* calculate dip angle (range of 0 (down) to 180 (up)) */
    	dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]=atan2(sqrt(grad1 * grad1 + grad2 * grad2), -grad3) / cRPD;
    	/* calculate azimuth angle (0 to 360) */
    	azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = atan2(grad1, grad2) / cRPD;
    	if (azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] < 0.0)
        	azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] += 360.0;
    	}
    else /*2D version*/
    for(is=0;is<nshot;is++)
	for(i1=0;i1<n1;i1++) /*x*/
	for(i2=0;i2<n2;i2++) /*y*/
	for(i3=1;i3<n3-1;i3++) /*z*/
    	{
    	
        if(n1==1)/*y direction*/
        {
        if(i2>=1 && i2<n2-1)
    	grad2=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+(i2-1)*n1+i3*n1*n2+is*n1*n2*n3])/d2 + 
    		  (t[i1+(i2+1)*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d2)/2.0;/*y*/
        }
        
        if(n2==1)/*x direction*/
        {
        if(i1>=1 && i1<n1-1)
    	grad1=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1-1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d1 + 
    		  (t[i1+1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d1)/2.0;/*x*/
        }
    	
    	grad3=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+(i3-1)*n1*n2+is*n1*n2*n3])/d3 + 
    		  (t[i1+i2*n1+(i3+1)*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d3)/2.0;/*z*/
    	
    	/* calculate dip angle (range of 0 (down) to 180 (up)) */
//     	dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]=atan2(sqrt(grad1 * grad1 + grad2 * grad2), -grad3) / cRPD;
//     	/* calculate azimuth angle (0 to 360) */
//     	azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = atan2(grad1, grad2) / cRPD;
//     	if (azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] < 0.0)
//         	azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] += 360.0;
//         	
        
        /* calculate dip angle (range of 0 (down) to 180 (up)) */
//         dip = atan2(grady, -gradz) / cRPD;
        
        if(n1==1)/*y direction*/
        {
        if(i2>=1 && i2<n2-1)
        dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = atan2(grad2, grad3) / cRPD;
        }
        if(n2==1)/*x direction*/
        {
        if(i1>=1 && i1<n1-1)
        dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = atan2(grad1, grad3) / cRPD;
        }
        
        iflip = 0;
        if (dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] > 180.0) {
            dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] - 180.0;
            iflip = 1;
        } else if (dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] < 0.0) {
            dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = -dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3];
            iflip = 1;
        }
        /* calculate azimuth polarity (1 or -1) relative to pos Y dir */
        azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = iflip ? -1.0 : 1.0;
        	
    	}
    
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*n3*nshot*3;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	
	for(i=0;i<n1*n2*n3*nshot;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = t[i];
	for(i=0;i<n1*n2*n3*nshot;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+n1*n2*n3*nshot))) = dip[i];
	for(i=0;i<n1*n2*n3*nshot;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+n1*n2*n3*nshot*2))) = azim[i];
		
	return PyArray_Return(vecout);
	
}


static PyObject *eikonalc_multishots(PyObject *self, PyObject *args){

    /*Below is the input part*/
    float f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13,f14;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;

	PyArg_ParseTuple(args, "OOOOffffffiiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13, &f14);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p, verb;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v;
    float *x, *y, *z;
    bool plane[3];
    
	o1=f4;
	o2=f5;
	o3=f6;
	
	d1=f7;
	d2=f8;
	d3=f9;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	verb=f14; /*verbosity*/
    
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);

    nd=PyArray_NDIM(arr1);
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arr1);
    npy_intp *spxyz=PyArray_SHAPE(arrf1);
    nshot=*spxyz;

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 


    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;

	ndim = 3; 
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));
	

    n123 = n1*n2*n3;

	t = (float*)malloc(n123*nshot * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));

	x = (float*)malloc(nshot * sizeof(float));
	y = (float*)malloc(nshot * sizeof(float));
	z = (float*)malloc(nshot * sizeof(float));


    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d\n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }

	/*reading xyz*/
    for (i=0; i<nshot; i++)
    {
        s[i][0]=*((float*)PyArray_GETPTR1(arrf1,i));
        s[i][1]=*((float*)PyArray_GETPTR1(arrf2,i));
        s[i][2]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
	
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    for( is = 0; is < nshot; is++) {
	if(verb) printf("shot %d of %d;\n",is+1,nshot);
	fastmarch(t+is*n123,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);
    }
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*n3*nshot;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = t[i];
	
	return PyArray_Return(vecout);
	
}

static PyObject *eikonalc_multishots_angle(PyObject *self, PyObject *args){
	/*also calculate take off angle*/
	/* following CalcAnglesGradient() in GridLib.c */
	
    /*Below is the input part*/
    float f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13,f14;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;

	PyArg_ParseTuple(args, "OOOOffffffiiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13, &f14);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p, verb;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v, *dip, *azim;
    float *x, *y, *z;
    bool plane[3];
    
	o1=f4;
	o2=f5;
	o3=f6;
	
	d1=f7;
	d2=f8;
	d3=f9;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	verb=f14; /*verbosity*/
    
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);

    nd=PyArray_NDIM(arr1);
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arr1);
    npy_intp *spxyz=PyArray_SHAPE(arrf1);
    nshot=*spxyz;

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 


    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;

	ndim = 3; 
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));
	

    n123 = n1*n2*n3;

	t = (float*)malloc(n123*nshot * sizeof(float));
	dip = (float*)malloc(n123*nshot * sizeof(float));
	azim = (float*)malloc(n123*nshot * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));

	x = (float*)malloc(nshot * sizeof(float));
	y = (float*)malloc(nshot * sizeof(float));
	z = (float*)malloc(nshot * sizeof(float));


    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d\n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }

	/*reading xyz*/
    for (i=0; i<nshot; i++)
    {
        s[i][0]=*((float*)PyArray_GETPTR1(arrf1,i));
        s[i][1]=*((float*)PyArray_GETPTR1(arrf2,i));
        s[i][2]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
	
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    for( is = 0; is < nshot; is++) {
	if(verb) printf("shot %d of %d;\n",is+1,nshot);
	fastmarch(t+is*n123,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);
    }
    
    
    int i1,i2,i3;
    float grad1,grad2,grad3;
    float cPI,cRPD;
    
    cPI = 4. * atan(1.); /* PI */
    cRPD = cPI / 180.; /* radians per degree */
    
    /*initialization of takeoff angle*/
    for(i=0;i<n123*nshot;i++)
    {dip[i]=200;azim[i]=400;}
    
    
    for(is=0;is<nshot;is++)
	for(i1=1;i1<n1-1;i1++) /*x*/
	for(i2=1;i2<n2-1;i2++) /*y*/
	for(i3=1;i3<n3-1;i3++) /*z*/
    	{
    	/*reverse sign to get take-off angle*/
    	grad1=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1-1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d1 + 
    		  (t[i1+1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d1)/2.0;
    	
    	grad2=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+(i2-1)*n1+i3*n1*n2+is*n1*n2*n3])/d2 + 
    		  (t[i1+(i2+1)*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d2)/2.0;
    	
    	grad3=-((t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+(i3-1)*n1*n2+is*n1*n2*n3])/d3 + 
    		  (t[i1+i2*n1+(i3+1)*n1*n2+is*n1*n2*n3]-t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3])/d3)/2.0;
    	
    	/* calculate dip angle (range of 0 (down) to 180 (up)) */
    	dip[i1+i2*n1+i3*n1*n2+is*n1*n2*n3]=atan2(sqrt(grad1 * grad1 + grad2 * grad2), -grad3) / cRPD;
    	/* calculate azimuth angle (0 to 360) */
    	azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] = atan2(grad1, grad2) / cRPD;
    	if (azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] < 0.0)
        	azim[i1+i2*n1+i3*n1*n2+is*n1*n2*n3] += 360.0;
    	}
    
    
    
    
    /*Output are time, dip: 0 (down) to 180 (up), azimuth: 0 to 360*/
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*n3*nshot*3;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<n1*n2*n3*nshot;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = t[i];
	for(i=0;i<n1*n2*n3*nshot;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+n1*n2*n3*nshot))) = dip[i];
	for(i=0;i<n1*n2*n3*nshot;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+n1*n2*n3*nshot*2))) = azim[i];
	
	return PyArray_Return(vecout);
	
}

static PyObject *eikonalc_surf(PyObject *self, PyObject *args){

    /*Below is the input part*/
    float f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13,f14;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;

	PyArg_ParseTuple(args, "OOOOffffffiiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13,&f14);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p, verb;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v;
    float *x, *y, *z;
    bool plane[3];
    
	o1=f4;
	o2=f5;
	o3=f6;
	
	d1=f7;
	d2=f8;
	d3=f9;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	verb=f14;
    
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);

    nd=PyArray_NDIM(arr1);
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arr1);
    npy_intp *spxyz=PyArray_SHAPE(arrf1);
    nshot=*spxyz;

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 

    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;

	ndim = 3;
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));
	

    n123 = n1*n2*n3;

	t = (float*)malloc(n123*nshot * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));

	x = (float*)malloc(nshot * sizeof(float));
	y = (float*)malloc(nshot * sizeof(float));
	z = (float*)malloc(nshot * sizeof(float));


    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d\n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }

	/*reading xyz*/
    for (i=0; i<nshot; i++)
    {
        s[i][0]=*((float*)PyArray_GETPTR1(arrf1,i));
        s[i][1]=*((float*)PyArray_GETPTR1(arrf2,i));
        s[i][2]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
	
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    int i1,i2,i3;
    float *tt;
    tt = (float*)malloc(n1*n2*nshot * sizeof(float)); /*nx*ny*nshot*/
    for( is = 0; is < nshot; is++) {
	if(verb) printf("shot %d of %d;\n",is+1,nshot);
	fastmarch(t+is*n123,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);

	for(i1=0;i1<n1;i1++) /*x*/
		for(i2=0;i2<n2;i2++) /*y*/
			for(i3=0;i3<1;i3++) /*z*/
				tt[i1+i2*n1+i3*n1*n2+is*n1*n2]=t[i1+i2*n1+i3*n1*n2+is*n1*n2*n3];
				
    }
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*nshot;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = tt[i];
	return PyArray_Return(vecout);
	
	
	
}


static PyObject *eikonalc_oneshot_rtp(PyObject *self, PyObject *args){

    /*Below is the input part*/
    float f1,f2,f3,f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd;

	PyArg_ParseTuple(args, "Offfffffffiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v;
    float x, y, z;
    bool plane[3];
    
	x=f1;		   /*r*/
	y=f2/180*M_PI; /*t*/
	z=f3/180*M_PI; /*p*/ 
	
	o1=f4;
	o2=f5/180*M_PI;
	o3=f6/180*M_PI;
	
	d1=f7;
	d2=f8/180*M_PI;
	d3=f9/180*M_PI;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    /*
     * my code starts here
     */
    nd=PyArray_NDIM(arr1);
    npy_intp *sp=PyArray_SHAPE(arr1);

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 

    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;
	
    /* File with shot locations (n2=number of shots, n1=3) */

	nshot = 1;
	ndim = 3;
     
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));

	s[0][0]=x; /*r*/
	s[0][1]=y; /*t*/
	s[0][2]=z; /*p*/
	
    n123 = n1*n2*n3;

	t = (float*)malloc(n123 * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));
	

    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d \n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }
    
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    nshot=1;
    for( is = 0; is < nshot; is++) {
	fastmarch_rtp(t,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);
    }
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*n3;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = t[i];
	
	return PyArray_Return(vecout);
	
}

static PyObject *eikonalc_multishots_rtp(PyObject *self, PyObject *args){

    /*Below is the input part*/
    float f4,f5,f6,f7,f8,f9;
    int f10,f11,f12,f13,f14;
    
	/**initialize data input**/
    PyObject *arg1=NULL;
    PyObject *arr1=NULL;
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;

	PyArg_ParseTuple(args, "OOOOffffffiiiii", &arg1, &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13,&f14);

    int b1, b2, b3, n1, n2, n3, nshot, ndim, i, is,order,n123, *p, verb;
    float br1, br2, br3, o1, o2, o3, d1, d2, d3;
    float **s, *t, *v;
    float *x, *y, *z;
    bool plane[3];
    
	o1=f4;			/*r*/
	o2=f5/180*M_PI; /*t*/
	o3=f6/180*M_PI; /*p*/
	 
	d1=f7;
	d2=f8/180*M_PI;
	d3=f9/180*M_PI;
	
	n1=f10;
	n2=f11;
	n3=f12;
	
	order=f13;
	verb=f14;
    
    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);

    nd=PyArray_NDIM(arr1);
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arr1);
    npy_intp *spxyz=PyArray_SHAPE(arrf1);
    nshot=*spxyz;

	br1=d1;
	br2=d2;
	br3=d3;
	plane[2]=false;
	plane[1]=false;
	plane[0]=false;
	b1= plane[2]? n1: (int) (br1/d1+0.5); 
	b2= plane[1]? n2: (int) (br2/d2+0.5);
	b3= plane[0]? n3: (int) (br3/d3+0.5); 


    if( b1<1 ) b1=1;  
    if( b2<1 ) b2=1;  
    if( b3<1 ) b3=1;

	ndim = 3; 
    s = (float**)malloc(nshot * sizeof(float*));
    for (int i = 0; i < nshot; i++)
        s[i] = (float*)malloc(ndim * sizeof(float));
	

    n123 = n1*n2*n3;

	t = (float*)malloc(n123*nshot * sizeof(float));
	v = (float*)malloc(n123 * sizeof(float));
	p = (float*)malloc(n123 * sizeof(float));

	x = (float*)malloc(nshot * sizeof(float));
	y = (float*)malloc(nshot * sizeof(float));
	z = (float*)malloc(nshot * sizeof(float));


    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_model = %d\n", *sp, n123);
    	return NULL;
    }
    
    /*reading velocity*/
    for (i=0; i<n123; i++)
    {
        v[i]=*((float*)PyArray_GETPTR1(arr1,i));
        v[i] = 1./(v[i]*v[i]);
    }

	/*reading xyz*/
    for (i=0; i<nshot; i++)
    {
        s[i][0]=*((float*)PyArray_GETPTR1(arrf1,i));
        s[i][1]=*((float*)PyArray_GETPTR1(arrf2,i))/180*M_PI;
        s[i][2]=*((float*)PyArray_GETPTR1(arrf3,i))/180*M_PI;
    }
	
    fastmarch_init (n3,n2,n1);
 
    /* loop over shots */
    for( is = 0; is < nshot; is++) {
	if(verb) printf("shot %d of %d;\n",is+1,nshot);
	fastmarch_rtp(t+is*n123,v,p, plane,
		      n3,n2,n1,
		      o3,o2,o1,
		      d3,d2,d1,
		      s[is][2],s[is][1],s[is][0], 
		      b3,b2,b1,
		      order);
    }
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n1*n2*n3*nshot;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = t[i];
	
	return PyArray_Return(vecout);
	
}

// documentation for each functions.
static char eikonalc_document[] = "Document stuff for eikonal...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"eikonalc_oneshot", eikonalc_oneshot, METH_VARARGS, eikonalc_document},
  {"eikonalc_oneshot_angle", eikonalc_oneshot_angle, METH_VARARGS, eikonalc_document},
  {"eikonalc_multishots", eikonalc_multishots, METH_VARARGS, eikonalc_document},
  {"eikonalc_multishots_angle",eikonalc_multishots_angle, METH_VARARGS, eikonalc_document},
  {"eikonalc_surf", eikonalc_surf, METH_VARARGS, eikonalc_document},
  {"eikonalc_oneshot_rtp", eikonalc_oneshot_rtp, METH_VARARGS, eikonalc_document},
  {"eikonalc_multishots_rtp", eikonalc_multishots_rtp, METH_VARARGS, eikonalc_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef eikonalcModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "eikonalc",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_eikonalc(void){

//   return PyModule_Create(&eikonalModule);
  
    PyObject *module = PyModule_Create(&eikonalcModule);
    import_array();
    return module;
}
