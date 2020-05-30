
/*********************************************************************************/

extern void CreateInts (int **vint, int dim);

extern void RemoveInts (int **vint);

extern void InitInts (int *vint, int dim, int frst, int incr);
		
extern void CopyInts (int *src, int *dst, int dim); 

extern void CopyShiftInts (int *src, int *dst, int dim, int shft);
	
extern void TransformLengthtoHeader (int *vint, int dim);

extern void TransformHeadertoLength (int *vint, int dim);

extern void ComputeHeaderfromLength (int *len, int *head, int dim);

extern void ComputeLengthfromHeader (int *head, int *len, int dim);

extern int AddInts (int *vint, int dim);

/*********************************************************************************/

extern void CreateDoubles (double **vdbl, int dim);

extern void RemoveDoubles (double **vdbl); 

extern void InitDoubles (double *vdbl, int dim, double frst, double incr);
		
extern void InitRandDoubles (double *vdbl, int dim, double frst, double last);

extern void CopyDoubles (double *src, double *dst, int dim); 

extern void ScaleDoubles (double *vdbl, double scal, int dim);

extern double DotDoubles (double *vdbl1, double *vdbl2, int dim);

extern void VvecDoubles (double alfa, double *src1, double *src2, double beta, double *dst, int dim);

/*********************************************************************************/
