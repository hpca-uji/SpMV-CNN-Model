#ifndef SparseProductTip

#define SparseProductTip 1

typedef struct
	{
		int dim1, dim2;
		int *vptr;
		int *vpos;
		double *vval;
	} SparseMatrix, *ptr_SparseMatrix;

/*********************************************************************************/

// This routine creates a sparseMatrix from the next parameters
// * numR defines the number of rows
// * numC defines the number of columns
// * numE defines the number of nonzero elements
// * msr indicates if the MSR is the format used to the sparse matrix
// If msr is actived, numE doesn't include the diagonal elements
// The parameter index indicates if 0-indexing or 1-indexing is used.
extern void CreateSparseMatrix (ptr_SparseMatrix p_spr, int index, int numR, int numC, int numE, 
																	int msr);

// This routine liberates the memory related to matrix spr
extern void RemoveSparseMatrix (ptr_SparseMatrix spr);

void FPrintMatlabFSparseMatrix (FILE *f, ptr_SparseMatrix spr, int index,
                                int fI1, int fI2, int fD1, int fD2) ;

FILE *OpenFile (char *name, char *attr);

void WriteMatlabFSparseMatrix (char *filename, ptr_SparseMatrix spr, int index,
                                int fI1, int fI2, int fD1, int fD2) ;
/*********************************************************************************/

// This routine creates de sparse matrix dst from the symmetric matrix spr.
// The parameters indexS and indexD indicate, respectivaly, if 0-indexing or 1-indexing is used
// to store the sparse matrices.
extern void DesymmetrizeSparseMatrices (SparseMatrix src, int indexS, ptr_SparseMatrix dst, 
																					int indexD);

int TransposeSparseMatrix(SparseMatrix A, ptr_SparseMatrix B);
/*********************************************************************************/

extern int ReadMatrixHB (char *filename, ptr_SparseMatrix p_spr);

/*********************************************************************************/

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVector2 (SparseMatrix spr, int index, double *vec, double *res);

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVectorByRows (SparseMatrix spr, int index, double *vec, double *res);

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVectorByRows_OMP (SparseMatrix spr, int index, double *vec, double *res);

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVectorByRows_OMPSS (SparseMatrix spr, int index, double *vec, double *res);

/*********************************************************************************/

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVectorByCols (SparseMatrix spr, int index, double *vec, double *res);

extern void ProdSparseMatrixVectorByColsBlocks (SparseMatrix spr, int index, double *vec, double *res, int *rows, int ini, int fin);
extern void ProdSparseMatrixVectorByColsBlocks2 (SparseMatrix spr, int index, double *vec, double *res, int *rows, int ini, int fin);
// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVectorByCols_OMP (SparseMatrix spr, int index, double *vec, double *res);

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
extern void ProdSparseMatrixVectorByCols_OMPSS (SparseMatrix spr, int index, double *vec, double *res);

/*********************************************************************************/

#endif
