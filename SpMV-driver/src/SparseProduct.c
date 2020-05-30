#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
/// #include "InputOutput.h"
#include "ScalarVectors.h"
#include "hb_io.h"
#include "SparseProduct.h"

/*********************************************************************************/

// This routine creates a sparseMatrix from the next parameters
// * numR defines the number of rows
// * numC defines the number of columns
// * numE defines the number of nonzero elements
// * msr indicates if the MSR is the format used to the sparse matrix
// If msr is actived, numE doesn't include the diagonal elements
// The parameter index indicates if 0-indexing or 1-indexing is used.
void CreateSparseMatrix (ptr_SparseMatrix p_spr, int index, int numR, int numC, int numE, int msr) {
//	printf (" index = %d , numR = %d , numC = %d , numE = %d\n", index, numR, numC, numE);
	// The scalar components of the structure are initiated
	p_spr->dim1 = numR; p_spr->dim2 = numC; 
	// Only one malloc is made for the vectors of indices
	CreateInts (&(p_spr->vptr), numE+numR+1);
	// The first component of the vectors depends on the used format
	*(p_spr->vptr) = ((msr)? (numR+1): 0) + index;
	p_spr->vpos = p_spr->vptr + ((msr)? 0: (numR+1));
	// The number of nonzero elements depends on the format used
	CreateDoubles (&(p_spr->vval), numE+(numR+1)*msr);
}

// This routine liberates the memory related to matrix spr
void RemoveSparseMatrix (ptr_SparseMatrix spr) {
	// First the scalar are initiated
	spr->dim1 = -1; spr->dim2 = -1; 
	// The vectors are liberated
	RemoveInts (&(spr->vptr)); RemoveDoubles (&(spr->vval)); 
}

// This routine writes on the file f the contents of spr, such that the data
// could be read by MATLAB. The parameter index indicates if 0-indexing 
// or 1-indexing is used. The values fI1 and fI2 refer to the way in which
// the indices are printed, while the fD1 and fD2 are related to the values.
void FPrintMatlabFSparseMatrix (FILE *f, ptr_SparseMatrix spr, int index,
                                int fI1, int fI2, int fD1, int fD2) {
  char formI[10], formD[10];
  int numI = 80 / fI1, numD = 80 / fD1;
  int i, j, k;
  printf("FPrintMatlbaFSparseMatrix\n");
  if (spr->dim1 > 0) {
    if (fI2 > 0) sprintf (formI, "%%%d.%dd ", fI1, fI2);
    else         sprintf (formI, "%%%dd ", fI1);
    if (fD2 > 0) sprintf (formD, "%%%d.%de ", fD1, fD2);
    else         sprintf (formD, "%%%de ", fD1);

    // If the MSR format is used, first the diagonals are printed
    if (spr->vpos == spr->vptr) {
      fprintf (f, "d = zeros(%d,1); \n", spr->dim1);
      fprintf (f, "d = [ ...  \n");
      k = numD;
      for (i=0; i<spr->dim1; i++) {
        fprintf (f, formD, spr->vval[i]);
        if ((--k) == 0) { fprintf (f, " ... \n"); k = numD; }
      }
      fprintf (f, "];\n");
    }
    // Then, the pointers to the sparse vectors are printed
    fprintf (f, "vptr = zeros(%d,1);  \n", spr->dim1+1);
    fprintf (f, "vptr = [ ... \n");
    k = numI;
    for (i=0; i<=spr->dim1; i++) {
      fprintf (f, formI, spr->vptr[i]);
      if ((--k) == 0) { fprintf (f, " ... \n"); k = numI; }
    }
    fprintf (f, "] + %d;\n", (1-index));
    // After, the rows of the nonzeos are printed
    fprintf (f, "rows = zeros(%d,1);  \n", spr->vptr[spr->dim1]-index);
    fprintf (f, "rows = [ ... \n");
    k = numI;
    int elem = 0; //Maria
    for (i=0; i<spr->dim1; i++) {
      for (j = spr->vptr[i]; j<spr->vptr[i+1]; j++) {
        fprintf (f, formI, i+index);
	elem++;
        if ((--k) == 0) { fprintf (f, " ... \n"); k = numI; }
      }
    }
    printf("Print rows --- elems: %d\n", elem);
    fprintf (f, "] + %d;\n", (1-index));
    // Before, the columns of the nonzeros are printed
    fprintf (f, "cols = zeros(%d,1);  \n", spr->vptr[spr->dim1]-index);
    fprintf (f, "cols = [ ... \n");
    k = numI;
    for (i=0; i<spr->vptr[spr->dim1]-index; i++) {
      fprintf (f, formI, spr->vpos[i]);
      if ((--k) == 0) { fprintf (f, " ... \n"); k = numI; }
    }
    fprintf (f, "] + %d;\n", (1-index));
    // Finally, the values of the nonzeros are printed
    fprintf (f, "vals = zeros(%d,1);  \n", spr->vptr[spr->dim1]-index);
    fprintf (f, "vals = [ ... \n");
    k = numD;
    for (i=0; i<spr->vptr[spr->dim1]-index; i++) {
      fprintf (f, formD, spr->vval[i]);
      if ((--k) == 0) { fprintf (f, " ... \n"); k = numD; }
    }
    fprintf (f, "];\n");
//    fprintf (f, "SpMat = sparse (rows+%d, cols+%d, vals);\n", 1-index, 1-index);
    fprintf (f, "SpMat = sparse (rows, cols, vals);\n");
    if (spr->vpos == spr->vptr) {
      fprintf (f, "SpMat = SpMat + diag (d);\n");
    }
  }
}

FILE *OpenFile (char *name, char *attr)
  {
    FILE *fich;
//    printf ("Opening file %s\n", name);
    if ((fich = fopen (name, attr)) == NULL)
      { printf ("File %s not exists \n", name); exit(1); }
    return fich;
  }

// This routine writes on the file filename the contents of spr, such that the data
// could be read by MATLAB. The parameter index indicates if 0-indexing 
// or 1-indexing is used. The values fI1 and fI2 refer to the way in which
// the indices are printed, while the fD1 and fD2 are related to the values.
void WriteMatlabFSparseMatrix (char *filename, ptr_SparseMatrix spr, int index,
                                int fI1, int fI2, int fD1, int fD2) {
  FILE *f = NULL;

  printf ("Writing %s of size (%d,%d)\n", filename, spr->dim1, spr->dim2);
//  printf ("Writing %s of size \n", filename);
//  printf ("dim1 = %d\n", spr.dim1);
//  printf ("dim2 = %d\n", spr.dim2);
  printf ("nnz = %d\n", spr->vptr[spr->dim1]-index);
  f = OpenFile (filename, "w");
  FPrintMatlabFSparseMatrix (f, spr, index, fI1, fI2, fD1, fD2);
  fclose (f);
  printf ("Written %s\n", filename);

}



/*********************************************************************************/

// This routine creates de sparse matrix dst from the symmetric matrix spr.
// The parameters indexS and indexD indicate, respectivaly, if 0-indexing or 1-indexing is used
// to store the sparse matrices.
void DesymmetrizeSparseMatrices (SparseMatrix src, int indexS, ptr_SparseMatrix dst, int indexD) {
	int n = src.dim1, nnz = 0;
	int *sizes = NULL;
	int *pp1 = NULL, *pp2 = NULL, *pp3 = NULL, *pp4 = NULL, *pp5 = NULL;
	int i, j, dim, indexDS = indexD - indexS;
	double *pd3 = NULL, *pd4 = NULL;

	// The vector sizes is created and initiated
	CreateInts (&sizes, n); InitInts (sizes, n, 0, 0);
	// This loop counts the number of elements in each row
	pp1 = src.vptr; pp3 = src.vpos + *pp1 - indexS;
	pp2 = pp1 + 1 ; pp4 = sizes - indexS;
	for (i=indexS; i<(n+indexS); i++) {
		// The size of the corresponding row is accumulated
		dim = (*pp2 - *pp1); pp4[i] += dim;
		// Now each component of the row is analyzed
		for (j=0; j<dim; j++) {
			// The nondiagonals elements define another element in the graph
			if (*pp3 != i) pp4[*pp3]++;
			pp3++;
		}
		pp1 = pp2++; 
	}
	
	// Compute the number of nonzeros of the new sparse matrix
	nnz = AddInts (sizes, n);
	// Create the new sparse matrix
	CreateSparseMatrix (dst, indexD, n, n, nnz, 0);
	// Fill the vector of pointers
	CopyInts (sizes, (dst->vptr) + 1, n);
	dst->vptr[0] = indexD; TransformLengthtoHeader (dst->vptr, n);
	// The vector sizes is initiated with the beginning of each row
	CopyInts (dst->vptr, sizes, n);
	// This loop fills the contents of vector vpos
	pp1 = src.vptr; pp3 = src.vpos + *pp1 - indexS; 
	pp2 = pp1 + 1 ; pp4 = dst->vpos - indexD; pp5 = sizes - indexS;
	pd3 = src.vval  + *pp1 - indexS; pd4 = dst->vval - indexD;
	for (i=indexS; i<(n+indexS); i++) {
		dim = (*pp2 - *pp1);
		for (j=0; j<dim; j++) {
			// The elements in the i-th row
			pp4[pp5[i]  ] = *pp3+indexDS; 
			pd4[pp5[i]++] = *pd3; 
			if (*pp3 != i) {
				// The nondiagonals elements define another element in the graph
				pp4[pp5[*pp3]  ] = i+indexDS;
				pd4[pp5[*pp3]++] = *pd3;
			}
			pp3++; pd3++;
		}
		pp1 = pp2++;
	}
	// The memory related to the vector sizes is liberated
	RemoveInts (&sizes);
}


int TransposeSparseMatrix(SparseMatrix A, ptr_SparseMatrix B)
{
/*----------------------------------------------------------------------
| Finds the transpose of a matrix stored in CSR format.
|
|-----------------------------------------------------------------------
| on entry:
|----------
| (A) = a matrix stored in CSR format.
|
| job    = integer to indicate whether to fill the values (job.eq.1)
|          of the matrix (B) or only the pattern.
|
| flag   = integer to indicate whether the target matrix has been filled
|          previously
|          0 - no filled
|          1 - filled
|
| ind    = integer buff of size at least nc
| on return:
| ----------
| (B) = the transpose of (A) stored in CSR format.
|
| integer value returned:
|             0   --> successful return.
|             1   --> memory allocation error.
|
|     code taken from ARMS of Yousef Saad.
|     adapted by Matthias Bollhoefer to fit with the data structures
|     of ILUPACK including the complex case
|---------------------------------------------------------------------*/
  int i, j, k, n, nnz, *ind;

  n=A.dim2;
  /*if (!flag) {
    B->dim1=A.dim2;
    B->dim2=A.dim1;
  }*/
//  nnz=A.vptr[A.dim1]-1;
  nnz=A.vptr[A.dim1]; //MARIA
  printf("A.dim1: %d, A.dim2:%d, n: %d, nnz: %d\n", A.dim1, A.dim2, n, nnz);
  CreateSparseMatrix (B, 0, A.dim2, A.dim1, nnz, 0);
  /* compute the length of each column of A */
  //if (!flag) {
    // B->vptr=(int *)malloc((n+1)*sizeof(int));
    // B->vpos=(int *)malloc(nnz  *sizeof(int));
    // if (job==1) 
    //    B->vval=(double *)malloc(nnz*sizeof(double));
    // else
    //    B->vval=NULL;
     
     // counter for the number of nonzeros for every row of B
     ind=B->vptr;
     for (i=0; i<=n; i++)
         ind[i] = 0;

     // run through A in order to find the nnz for each column of A
     // do not adjust ind to fit with FORTRAN indexing!
     for (j=0; j<nnz; j++) {
         ind[A.vpos[j]]++;
//	 printf("ind[%d]: %d\n", A.vpos[j], ind[A.vpos[j]]);
     }
     // change ind such that ind[i] holds the start of column i
     // ind[0] has not been used due to different indexing in FORTRAN
     //ind[0]=1;
     //ind[0]=0; //Maria
     for (i=0; i<n; i++) {
         ind[i+1]+=ind[i];

//	 printf("ind[%d]: %d\n", i+1, ind[i+1]);
     }
//     printf("**********************");
     for (i=n; i>0; i--) {
      ind[i]=ind[i-1];
//      printf("ind[%d]: %d\n", i, ind[i]);
    }
    ind[0]=0; //Maria*/
//  } // end if
//  else
//     ind=B->vptr;


  /*--------------------  now do the actual copying  */
  //for (i=0; i<n; i++) { 
  for (i=0; i<A.dim1; i++) {  //MARIA
    // copy indices
    for (j=A.vptr[i]; j<A.vptr[i+1]; j++) {
        // current column index of row i in FORTRAN style
        //k=A.vpos[j-1]-1;
        k=A.vpos[j]; //MARIA
//	printf("i: %d, j: %d, k: %d, ind[%d]: %d\n", i, j, k, k, ind[k]);
	//B->vpos[ind[k]-1]=i+1;
	B->vpos[ind[k]]=i; //MARIA
	//B->vval[ind[k]-1]=A.vval[j-1];
	B->vval[ind[k]]=A.vval[j]; //MARIA
	// advance pointer
	ind[k]++;  //MARIA
    } // end for j
  } // end for i

  // shift ind back
  for (i=n; i>0; i--) {
      ind[i]=ind[i-1];
  //    printf("ind[%d]: %d\n", i, ind[i]);
  }
  //ind[0]=1;
  ind[0]=0; //MARIA
 // printf("B->vptr[%d]: %d, B->vpos[%d]: %d\n", B->dim1, B->vptr[B->dim1], nnz, B->vpos[nnz]);

  return 0;
}

/*********************************************************************************/

int ReadMatrixHB (char *filename, ptr_SparseMatrix p_spr) {
  int *colptr = NULL;
  double *exact = NULL;
  double *guess = NULL;
  int indcrd;
  char *indfmt = NULL;
  FILE *input;
  char *key = NULL;
  char *mxtype = NULL;
  int ncol;
  int neltvl;
  int nnzero;
  int nrhs;
  int nrhsix;
  int nrow;
  int ptrcrd;
  char *ptrfmt = NULL;
  int rhscrd;
  char *rhsfmt = NULL;
  int *rhsind = NULL;
  int *rhsptr = NULL;
  char *rhstyp = NULL;
  double *rhsval = NULL;
  double *rhsvec = NULL;
  int *rowind = NULL;
  char *title = NULL;
  int totcrd;
  int valcrd;
  char *valfmt = NULL;
  double *values = NULL;

	printf ("\nTEST09\n");
	printf ("  HB_FILE_READ reads all the data in an HB file.\n");
	printf ("  HB_FILE_MODULE is the module that stores the data.\n");

	input = fopen (filename, "r");
	if ( !input ) {
		printf ("\n TEST09 - Warning!\n Error opening the file %s .\n", filename);
		return -1;
	}

	hb_file_read ( input, &title, &key, &totcrd, &ptrcrd, &indcrd,
									&valcrd, &rhscrd, &mxtype, &nrow, &ncol, &nnzero, &neltvl,
									&ptrfmt, &indfmt, &valfmt, &rhsfmt, &rhstyp, &nrhs, &nrhsix,
									&colptr, &rowind, &values, &rhsval, &rhsptr, &rhsind, &rhsvec,
									&guess, &exact );
	fclose (input);

	// Conversion Fortran to C
	CopyShiftInts (colptr, colptr, nrow+1, -1);
	CopyShiftInts (rowind, rowind, nnzero, -1);

	printf("nrow: %d, ncol: %d, colptr[%d]: %d, rowind[%d]: %d\n", nrow, ncol, nrow, colptr[nrow], nnzero, rowind[nnzero]);
	//  Data assignment
//	p_spr->dim1 = nrow  ; p_spr->dim2 = ncol  ; 
	p_spr->dim1 = ncol  ; p_spr->dim2 = nrow  ; 
	p_spr->vptr = colptr; p_spr->vpos = rowind; p_spr->vval = values; 
	printf("p_spr->dim1: %d, p_spr->dim2: %d\n", p_spr->dim1, p_spr->dim2);
	//  Memory liberation
	free (exact ); free (guess ); free (indfmt);
	free (key   ); free (mxtype); free (ptrfmt);
	free (rhsfmt); free (rhsind); free (rhsptr);
	free (rhstyp); free (rhsval); free (rhsvec);
	free (title ); free (valfmt);
	
	return 0;
}

/*********************************************************************************/

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
void ProdSparseMatrixVector2 (SparseMatrix spr, int index, double *vec, double *res) {
	int i, j;
	int *pp1 = spr.vptr, *pp2 = pp1+1, *pi1 = spr.vpos + *pp1 - index;
	double aux, *pvec = vec - index, *pd2 = res;
	double *pd1 = spr.vval + *pp1 - index;

	// If the MSR format is used, first the diagonal has to be processed
	if (spr.vptr == spr.vpos)
		VvecDoubles (1.0, spr.vval, vec, 1.0, res, spr.dim1);

	for (i=0; i<spr.dim1; i++) {
		// The dot product between the row i and the vector vec is computed
		aux = 0.0;
		for (j=*pp1; j<*pp2; j++)
			aux += *(pd1++) * pvec[*(pi1++)];
//		for (j=spr.vptr[i]; j<spr.vptr[i+1]; j++)
//			aux += spr.vval[j] * pvec[spr.vpos[j]];
		// Accumulate the obtained value on the result
		*(pd2++) += aux; pp1 = pp2++;
	}
}

// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
void ProdSparseMatrixVectorByRows (SparseMatrix spr, int index, double *vec, double *res) {
	int i, j, dim = spr.dim1;
	int *pp1 = spr.vptr, *pi1 = spr.vpos + *pp1 - index;
	double aux, *pvec = vec + *pp1 - index;
	double *pd1 = spr.vval + *pp1 - index;

	// Process all the rows of the matrix
	for (i=0; i<dim; i++) {
		// The dot product between the row i and the vector vec is computed
		aux = 0.0;
		for (j=pp1[i]; j<pp1[i+1]; j++)
			aux += pd1[j] * pvec[pi1[j]];
		// Accumulate the obtained value on the result
		res[i] += aux; 
	}
}

/*********************************************************************************/

/*// This routine computes the product { res += spr * vec }.
// The parameter index indicates if 0-indexing or 1-indexing is used,
void ProdSparseMatrixVectorByCols (SparseMatrix spr, int index, double *vec, double *res) {
	int i, j, dim = spr.dim1;
	int *pp1 = spr.vptr, *pi1 = spr.vpos + *pp1 - index;
	double aux, *pres = res + *pp1 - index;
	double *pd1 = spr.vval + *pp1 - index;

	// Process all the columns of the matrix
	for (i=0; i<dim; i++) {
		// The result is scaled by the column i and the scalar vec[i]
		aux = vec[i];
		for (j=pp1[i]; j<pp1[i+1]; j

	// Process all the non-zeros of the block
	for ( i = ini; i < fin && i < spr.vptr[spr.dim1]; i++ ) {
		res[rows[i]] += spr.vval[i] * vec[spr.vpos[i]];
//		fprintf(fp, "%d\t", col);
	}
}*/

void ProdSparseMatrixVectorByColsBlocks2 (SparseMatrix spr, int index, double *vec, double *res, int *rows, int ini, int fin) {
	int i, prv = spr.dim1;
        double aux = 0.0;

	// Process all the non-zeros of the block
	for ( i = ini; i < fin; i++ ) {
		// accumulates only when vpos[i-1]>col meaning that it went to the next row
		// if ( spr.vpos[i] < ant ) {
		if ( rows[i] > prv ) {
                    res[rows[i-1]] += aux;
		    aux = 0.0;
		}
		aux += spr.vval[i] * vec[spr.vpos[i]];
                prv = rows[i];
	//	printf("vval[%d] = %lu ; vpos[%d] = %lu ; vec[vpos[%d]] = %lu\n", i, (unsigned long)&(spr.vval[i]), i, (unsigned long)&(spr.vpos[i]), i, (unsigned long)&(vec[spr.vpos[i]]));
	}
	//printf("vval[0]   = %lu ; vpos[0]   = %lu ; vec[0] = %lu\n", (unsigned long)&(spr.vval[0]),                  (unsigned long)&(spr.vpos[spr.vptr[0]]),        (unsigned long)&(vec[0]));
	//printf("vval[nnz] = %lu ; vpos[nnz] = %lu ; vec[n] = %lu\n", (unsigned long)&(spr.vval[spr.vptr[spr.dim1]]), (unsigned long)&(spr.vpos[spr.vptr[spr.dim1]]), (unsigned long)&(vec[spr.vpos[spr.dim1]]));
    // final accumulation
    res[prv] += aux;
}
