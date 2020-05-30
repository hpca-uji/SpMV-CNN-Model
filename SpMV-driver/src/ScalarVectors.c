#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ScalarVectors.h>

/*********************************************************************************/

void CreateInts (int **vint, int dim) {
	if ((*vint = (int *) malloc (sizeof(int)*dim)) == NULL)
		{ printf ("Memory Error (CreateInts(%d))\n", dim); exit (1); }
}

void RemoveInts (int **vint) { 
	if (*vint != NULL) free (*vint); *vint = NULL; 
}

void InitInts (int *vint, int dim, int frst, int incr) {
	int i, *p1 = vint, num = frst;

	for (i=0; i<dim; i++) 
		{ *(p1++) = num; num += incr; }
}
		
void CopyInts (int *src, int *dst, int dim) { 
	memmove (dst, src, sizeof(int) * dim);
}

void CopyShiftInts (int *src, int *dst, int dim, int shft) {
	int i, *p1 = src, *p2 = dst;

	if (shft == 0)
		CopyInts (src, dst, dim);
	else
		for (i=0; i<dim; i++)
			*(p2++) = *(p1++) + shft;
}
	
void TransformLengthtoHeader (int *vint, int dim) {
	int i, *pi = vint; 

	for (i=0; i<dim; i++) { *(pi+1) += *pi; pi++; }
}

void TransformHeadertoLength (int *vint, int dim) {
	int i, *pi = vint+dim; 

	for (i=dim; i>0; i--) { *(pi) -= *(pi-1); pi--; }
}

void ComputeHeaderfromLength (int *len, int *head, int dim) {
	int i, *pi1 = len, *pi2 = head; 

	for (i=0; i<dim; i++) { *(pi2+1) = (*pi2) +(*(pi1++)); pi2++; }
}

void ComputeLengthfromHeader (int *head, int *len, int dim) {
	int i, *pi1 = head, *pi2 = len; 

	for (i=0; i<dim; i++) { *(pi2++) = (*(pi1+1)) -(*pi1); pi1++; }
}

int AddInts (int *vint, int dim) {
	int i, *pi = vint, aux = 0;

	for (i=0; i<dim; i++) { 
		aux += *pi; pi++; 
	}

	return aux;
}

/*********************************************************************************/

void CreateDoubles (double **vdbl, int dim) {
	if ((*vdbl = (double *) malloc (sizeof(double)*dim)) == NULL)
		{ printf ("Memory Error (CreateDoubles(%d))\n", dim); exit (1); }
}

void RemoveDoubles (double **vdbl) { 
	if (*vdbl != NULL) free (*vdbl); *vdbl = NULL; 
}

void InitDoubles (double *vdbl, int dim, double frst, double incr) {
	int i; 
	double *pd = vdbl, num = frst;

	for (i=0; i<dim; i++) 
		{ *(pd++) = num; num += incr; }
}
		
void InitRandDoubles (double *vdbl, int dim, double frst, double last) {
	int i; 
	double *pd = vdbl, size = last - frst;

	for (i=0; i<dim; i++) 
		{ *(pd++) = frst + (size * (rand() / (RAND_MAX + 1.0))); }
}
		
void CopyDoubles (double *src, double *dst, int dim) { 
	memmove (dst, src, sizeof(double) * dim);
}

void ScaleDoubles (double *vdbl, double scal, int dim) {
	int i; 
	double *pd = vdbl;

	for (i=0; i<dim; i++) 
		*(pd++) *= scal;
}

double DotDoubles (double *vdbl1, double *vdbl2, int dim) {
	int i; 
	double *pd1 = vdbl1, *pd2 = vdbl2, res = 0.0;

	for (i=0; i<dim; i++) 
		res += (*(pd2++)) * (*(pd1++));

	return res;
}

void VvecDoubles (double alfa, double *src1, double *src2, double beta, double *dst, int dim) {
	int i;

	for (i=0; i<dim; i++) { 
		*dst = (beta * *dst) + (alfa * *(src1++) * *(src2++)); dst++; 
	}
}


/*********************************************************************************/
