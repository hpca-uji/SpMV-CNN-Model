#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <papi.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <reloj.h>
#include <hdf5.h>
#include <ScalarVectors.h>
#include <hb_io.h>
#include <SparseProduct.h>

#define NUM_EVENTS 2

char events[NUM_EVENTS][BUFSIZ]={
  "PACKAGE_ENERGY:PACKAGE0",
  "DRAM_ENERGY:PACKAGE0",
};

char filenames[NUM_EVENTS+2][BUFSIZ]={
  "results.PACKAGE_ENERGY_PACKAGE0",
  "results.DRAM_ENERGY_PACKAGE0",
};

FILE *fff[NUM_EVENTS+2];

#define ENERGY 1

/*********************************************************************************/

int extrae_argumentos(char *orig, char *delim, char ***args)
{
	char *tmp;
	int num=0;
	/* Reservamos memoria para copiar la cadena ... pero la memoria justa */
	char *str= malloc(strlen(orig)+1);
	char **aargs;
	
	strcpy(str, orig);
	
	aargs=malloc(sizeof(char**));

	tmp=strtok(str, delim);
	do
	{
		aargs[num]=malloc(sizeof(char*));

		/*       strcpy(aargs[num], tmp); */
		aargs[num]=tmp;
		num++;
		
		/* Reservamos memoria para una palabra más */
		aargs=realloc(aargs, sizeof(char**)*(num+1));

		/* Extraemos la siguiente palabra */
		tmp=strtok(NULL, delim);
	} while (tmp!=NULL);

	*args=aargs;
	return num;
}

int main (int argc, char *argv[]) {
  int i, j, dim = 0, dim2=0, nnz; 
  int reps = atoi(argv[2]);
  int t_block; 
  int min_block = atoi(argv[3]);         // Minimum block size
  int max_block = atoi(argv[4]);	 // Maximum block size
  int diff_blocks = atoi(argv[5]);	 // Difference between block sizes
  int base = atoi(argv[6]);		 // The starting nnz of the matrix
  int frec[1] = {atoi(argv[7])};	 // The frequency
  int *rows = NULL;
  int nargs;
  int ini = base, fin, b = 0, num_blocks, max_num_blocks;
  char file[40];
  char **args;
  double norm = 0.0;
  double *vec = NULL, *sol2 = NULL;
  double te1, tu1,te2, tu2, teI1, teI2, tuI1, tuI2;
  long long s, e;
  int retval;
  double t_total = 0.0, t_avg = 0.0, t_sum = 0.0;
  double t_total_full = 0.0, e_total_full = 0.0; 
  int bl = ((max_block - min_block)/diff_blocks)+1;
  double *block_times = NULL;
  
  //ENERGY MEASUREMENTS
#ifdef ENERGY  
  int EventSet = PAPI_NULL;
  long long values[NUM_EVENTS];
  double *block_energy = NULL;
  double *block_package_energy = NULL;
  double *block_dram_energy = NULL;
  double *block_pp0_energy = NULL;
  double *block_uncore_energy = NULL;
  double e_total = 0.0, energy, e_avg = 0.0, e_sum = 0.0;
  double e_total_pkg, e_total_dram, e_total_pp0, e_total_uncore;
  double energy_pkg, energy_dram; 
#endif

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if ( retval != PAPI_VER_CURRENT ) {
    fprintf(stderr,"PAPI_library_init failed  retval: %d, PAPI_VER_CURRENT: %d\n", retval, PAPI_VER_CURRENT);
  }

  // Matrix declaration
  SparseMatrix mat = {0, 0, NULL, NULL, NULL}, sym = {0, 0, NULL, NULL, NULL};


  // Create output filename
  nargs=extrae_argumentos(argv[1], ".", &args);
  nargs=extrae_argumentos(args[nargs-2], "/", &args);
  sprintf(file, "output_%s_%s.h5", args[nargs-1], argv[7]);
  printf("%s\n", file);

  // Creating the matrix
  ReadMatrixHB (argv[1], &sym);
  if (argv[8] == 1)  // If 1 the matrix is symmetric. If 0 the matrix is no-symmetric.
  	DesymmetrizeSparseMatrices (sym, 0, &mat, 0); // For symmetric matrices
  else
	TransposeSparseMatrix(sym, &mat); //For no-symmetric matrices
  RemoveSparseMatrix (&sym);

  dim = mat.dim1;
  dim2 = mat.dim2;
  nnz = mat.vptr[dim];

  max_num_blocks= (int)(ceil(nnz/(float)(min_block)));
  // Creating the vectors
  CreateDoubles (&vec , dim2);
  CreateDoubles (&sol2, dim);
  InitRandDoubles (vec, dim2, -1.0, 1.0);
  CreateDoubles (&block_times, max_num_blocks);
  CreateInts (&rows, nnz);
#ifdef ENERGY
  CreateDoubles (&block_energy, max_num_blocks);
  CreateDoubles (&block_package_energy, max_num_blocks);
  CreateDoubles (&block_dram_energy, max_num_blocks);
  CreateDoubles (&block_pp0_energy, max_num_blocks);
  CreateDoubles (&block_uncore_energy, max_num_blocks);
#endif

  // Init vector of rows
  int count = 0;
  for ( i = 0; i < dim; ++i ) {
    for ( j = mat.vptr[i]; j < mat.vptr[i+1]; j++ ) {
  	  rows[j] = i;
		  count++;
    }
  }
	printf("count: %d, dim = %d, dim2 = %d, reps = %d, nnz = %d, bl = %d\n", count, dim, dim2, reps, nnz, bl);

  // Compute global time with the by-default routine to heat the machine
  printf("START HEATING ....\n");
	InitDoubles (sol2, dim, 0.0, 0.0);
  int end = 0;
  int time = 0;
	reloj (&te1, &tu1);
	while(end != 1) {
	 	ProdSparseMatrixVectorByRows (mat, 0, vec, sol2);
		reloj (&te2, &tu2);
    time += te2-te1;
    if(time > 180)
			end = 1; 
	}
  printf("END HEATING ....\n");
  
  // HDF5 datasets
  hid_t file_h5, mat_blocks_h5, block_times_h5, dim_h5, dim_frec, times_h5, dsetF, dset1, dset2, dset3, dset4;
  herr_t status;
  hsize_t dims[1] = {nnz}, dims2[1] = {3}, dims4[1] ={bl+1}, dimsF[1]={1};
  int headers[3] = {dim2, dim, nnz};

  file_h5 = H5Fcreate (file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  mat_blocks_h5= H5Screate_simple (1, dims, NULL);
  dim_h5= H5Screate_simple (1, dims2, NULL);
  dim_frec= H5Screate_simple (1, dimsF, NULL);
  times_h5= H5Screate_simple (1, dims4, NULL);
#ifdef ENERGY
  hid_t block_energy_h5, block_ePkg_h5, block_eDram_h5, energy_h5, energyPackage_h5, energyDram_h5, dset5, dset6, dset7, dset8; 
  energy_h5= H5Screate_simple (1, dims4, NULL);
  energyPackage_h5= H5Screate_simple (1, dims4, NULL);
  energyDram_h5= H5Screate_simple (1, dims4, NULL);
#endif

  dsetF = H5Dcreate (file_h5, "FREC", H5T_NATIVE_INT, dim_frec, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  dset1 = H5Dcreate (file_h5, "HEAD", H5T_NATIVE_INT, dim_h5, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  dset2 = H5Dcreate (file_h5, "MB", H5T_NATIVE_INT, mat_blocks_h5, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  status = H5Dwrite (dsetF, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, 
                     H5P_DEFAULT, frec);
  status = H5Dwrite (dset1, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, 
                     H5P_DEFAULT, headers);
  status = H5Dwrite (dset2, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, mat.vpos);

#ifdef ENERGY
  /* Create EventSet */
  retval = PAPI_create_eventset( &EventSet );
  if (retval != PAPI_OK) {
      fprintf(stderr,"Error creating eventset! retval: %d, PAPI_OK: %d\n", retval, PAPI_OK);
  }
  for(i=0;i<NUM_EVENTS;i++) {
    retval = PAPI_add_named_event( EventSet, events[i] );
    if (retval != PAPI_OK) {
	    fprintf(stderr,"Error adding event %s, retval: %d, PAPI_OK: %d\n",events[i], retval, PAPI_OK);
	  }
  }
#endif
  
  // Compute global time with the by-blocks routine
  printf("By-blocks routine (complete)\n");
  InitDoubles (sol2, dim, 0.0, 0.0);
  reloj (&te1, &tu1);
#ifdef ENERGY
  retval = PAPI_start( EventSet);
  if (retval != PAPI_OK) {
  	fprintf(stderr,"PAPI_start() failed\n");
	  exit(1);
  }
#endif
  s = PAPI_get_real_cyc();
  for (i=0; i<reps; i++) {
     ProdSparseMatrixVectorByColsBlocks2 (mat, 0, vec, sol2, rows, 0, nnz);
  }
  e = PAPI_get_real_cyc();
#ifdef ENERGY
  retval = PAPI_stop( EventSet, values);
  if (retval != PAPI_OK) {
  	fprintf(stderr, "PAPI_start() failed\n");
  }
  energy_pkg = (double)values[0]; 
  energy_dram = (double)values[1]; 
  energy = energy_pkg + energy_dram;  //PACKAGE + DRAM
  e_total_full = (energy/reps)/nnz;  // Energy consumed by non-zero element when executing the whole matrix
#endif
  reloj (&te2, &tu2);
  norm = DotDoubles (sol2, sol2, dim);
  t_total_full = (((e-s)/reps)/((double)frec[0]/1000000.0))/nnz; // Time consumed by non-zero element when executing the whole matrix
  printf (" Norm = %20.15e , t_total_nnz: %20.15f e_total_nnz = %20.15e\n frec: %d\n", norm, t_total_full, e_total_full, frec[0]);
  
  double t[bl+1];
  t[bl] = t_total_full;
  dset4 = H5Dcreate (file_h5, "TOT_TIME", H5T_NATIVE_DOUBLE, times_h5, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#ifdef ENERGY
  double en[bl+1];
  double enPkg[bl+1];
  double enDram[bl+1];
  en[bl] = e_total_full;
  enPkg[bl] = (energy_pkg/reps)/nnz; 
  enDram[bl] = (energy_dram/reps)/nnz; 
  dset6 = H5Dcreate (file_h5, "TOT_ENERGY", H5T_NATIVE_DOUBLE, energy_h5, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  dset7 = H5Dcreate (file_h5, "TOT_ENERGY_PKG", H5T_NATIVE_DOUBLE, energyPackage_h5, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  dset8 = H5Dcreate (file_h5, "TOT_ENERGY_DRAM", H5T_NATIVE_DOUBLE, energyDram_h5, 
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
  // Compute global time with the by-blocks routine
  printf("By-blocks routine (blocks)\n");
  int cont = 0;
  for (t_block = min_block; t_block <= max_block; t_block+=diff_blocks) {
    // Compute number of blocks
    num_blocks= (int)(ceil(nnz/(float)t_block));
    printf("Number of blocks: %d\n", num_blocks);
    b = 0;
    t_total = 0.0;
#ifdef ENERGY
    e_total = 0.0;
    e_total_pkg = 0.0;
    e_total_dram = 0.0;
    e_total_pp0 = 0.0;
    e_total_uncore = 0.0;
#endif
    ini = base;
    fin = ini + t_block;
    InitDoubles (sol2, dim, 0.0, 0.0);
    while (ini < nnz) {
      reloj (&teI1, &tuI1);
#ifdef ENERGY
      retval = PAPI_start( EventSet);
      if (retval != PAPI_OK) {
         fprintf(stderr,"PAPI_start() failed\n");
	       exit(1);
      }
#endif
      s = PAPI_get_real_cyc();
      for (i=0; i<reps; i++) {
        ProdSparseMatrixVectorByColsBlocks2 (mat, 0, vec, sol2, rows, ini, fin);
      }
      e = PAPI_get_real_cyc();
#ifdef ENERGY
      retval = PAPI_stop( EventSet, values);
      if (retval != PAPI_OK) {
         fprintf(stderr, "PAPI_start() failed\n");
      }
      energy_pkg = (double)values[0]; 
      energy_dram = (double)values[1]; 
      energy = energy_pkg + energy_dram;  //PACKAGE + DRAM
#endif
      reloj (&teI2, &tuI2);
      block_times[b]= (((e-s)/reps)/((double)frec[0]/1000000)) / (fin - ini);
      t_total += block_times[b];
#ifdef ENERGY 	    
      block_energy[b]= (energy/reps) / (fin - ini);
      block_package_energy[b] = (energy_pkg/reps) / (fin - ini);
      block_dram_energy[b] = (energy_dram/reps) / (fin - ini);
      e_total += block_energy[b];   // Total energy consumed adding all the blocks 
      e_total_pkg += block_package_energy[b]; // Package energy consumed adding all the blocks 
      e_total_dram += block_dram_energy[b]; // Dram energy consumed adding all the blocks 
#endif
      ini = fin; 
      fin = (ini + t_block) < nnz ? (ini + t_block) : nnz;
      b++;
    }
    norm = DotDoubles (sol2, sol2, dim);
    t_avg = t_total/num_blocks, t_sum = 0;
    t[cont] = t_avg;
#ifdef ENERGY 	    
    e_avg = e_total/num_blocks, e_sum = 0;
    en[cont] = e_avg;
    enPkg[cont] = e_total_pkg/num_blocks;
    enDram[cont] = e_total_dram/num_blocks;
#endif
    printf("B: %d  Norm = %20.15e , t_total = %20.15f, t_nnz = %20.15f", b, norm, t_avg*nnz, t_avg);
#ifdef ENERGY 	    
    printf("B: %d  Norm = %20.15e , t_total = %20.15f, t_nnz = %20.15f, e_total = %20.15e, e_nnz: %20.15e\n", b, norm, t_avg*nnz, t_avg, e_avg*nnz, e_avg);

#endif
    cont++;

    // HDF5
    hsize_t dims3[1] = {b};
    char dataset[8];
    sprintf(dataset, "B_%d", t_block);
    block_times_h5= H5Screate_simple (1, dims3, NULL);

    dset3 = H5Dcreate (file_h5, dataset, H5T_NATIVE_DOUBLE, block_times_h5, 
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite (dset3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, block_times);
    status = H5Dclose (dset3);
    status = H5Sclose (block_times_h5);
#ifdef ENERGY    
      // TOTAL ENERGY
    char datasetE[8];
    sprintf(datasetE, "E_%d", t_block);
    block_energy_h5= H5Screate_simple (1, dims3, NULL);
    dset5 = H5Dcreate (file_h5, datasetE, H5T_NATIVE_DOUBLE, block_energy_h5, 
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite (dset5, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, block_energy);
    status = H5Dclose (dset5);
    status = H5Sclose (block_energy_h5);
  
    // PKG ENERGY
    char datasetEPKG[8];
    sprintf(datasetEPKG, "EPKG_%d", t_block);
    block_ePkg_h5= H5Screate_simple (1, dims3, NULL);
    dset5 = H5Dcreate (file_h5, datasetEPKG, H5T_NATIVE_DOUBLE, block_ePkg_h5, 
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite (dset5, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, block_package_energy);
    status = H5Dclose (dset5);
    status = H5Sclose (block_ePkg_h5);

    // DRAM ENERGY
    char datasetED[8];
    sprintf(datasetED, "EDRAM_%d", t_block);
    block_eDram_h5= H5Screate_simple (1, dims3, NULL);
    dset5 = H5Dcreate (file_h5, datasetED, H5T_NATIVE_DOUBLE, block_eDram_h5, 
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite (dset5, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                       H5P_DEFAULT, block_dram_energy);
    status = H5Dclose (dset5);
    status = H5Sclose (block_eDram_h5);
   
#endif
  }
  status = H5Dwrite (dset4, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                    H5P_DEFAULT, t);
#ifdef ENERGY
  status = H5Dwrite (dset6, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                    H5P_DEFAULT, en);
  status = H5Dwrite (dset7, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                    H5P_DEFAULT, enPkg);
  status = H5Dwrite (dset8, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                    H5P_DEFAULT, enDram);
#endif
  status = H5Dclose (dsetF);
  status = H5Dclose (dset1);
  status = H5Dclose (dset2);
  status = H5Dclose (dset4);
  status = H5Sclose (mat_blocks_h5);
  status = H5Sclose (times_h5);
  status = H5Sclose (dim_frec);
  status = H5Sclose (dim_h5);
  status = H5Fclose (file_h5);
#ifdef ENERGY
  status = H5Dclose (dset6);
  status = H5Dclose (dset7);
  status = H5Dclose (dset8);
  status = H5Sclose (energy_h5);
  status = H5Sclose (energyPackage_h5);
  status = H5Sclose (energyDram_h5);
#endif
     
/***************************************/

// Freing memory
#ifdef ENERGY
  RemoveDoubles(&block_energy);
  RemoveDoubles(&block_package_energy);
  RemoveDoubles(&block_dram_energy);
#endif
  RemoveDoubles (&sol2); 
  RemoveDoubles (&vec);
  RemoveDoubles(&block_times);
  RemoveInts(&rows);
  RemoveSparseMatrix (&mat); 

  return 0;
}
