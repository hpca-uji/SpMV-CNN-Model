# ============================================================

CC = gcc

#MODIFY THE PATH
PAPI_HOME = /home/mvaya/papi_install
PAPI_INCLUDE = $(PAPI_HOME)/include
PAPI_LIBRARY = -L$(PAPI_HOME)/lib -lpapi -Wl,-rpath,$(PAPI_HOME)/lib

#MODIFY THE PATH
HDF5_HOME = /home/mvaya/hdf5
HDF5_INC = $(HDF5_HOME)/include
HDF5_LIB = -L$(HDF5_HOME)/lib   

CFLAGS = -Wall -I. -O1 -I$(HDF5_INC) -I$(PAPI_INCLUDE) -g 
CLINKER = gcc
LIBLIST = -lm -lc $(HDF5_LIB) -lhdf5 $(PAPI_LIBRARY) 
# ============================================================
AR = ar

RL = ranlib

# ============================================================

OBJS_CLOCK  = reloj.o 
OBJS_VECTOR = ScalarVectors.o
OBJS_SPARSE = hb_io.o SparseProduct.o

OBJS = $(OBJS_CLOCK) $(OBJS_VECTOR) $(OBJS_SPARSE) 

# ============================================================

default: libclock.a libvector.a libsparse.a driver

libshared.a : $(OBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

libclock.a : $(OBJS_CLOCK)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

libvector.a : $(OBJS_VECTOR)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

libsparse.a : $(OBJS_SPARSE)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

%.o: %.c
	$(CLINKER) -c -o $@ $< $(CFLAGS)


driver: $(OBJS) driver.o 
	$(CLINKER) $(LDFLAGS) -o $@ $^ $(LIBLIST)

# ============================================================

.c.o:
	echo compilando
	$(CC) $(CFLAGS) -c $*.c

clean:
	rm *.o

clear:
	rm *.o *.a

# ============================================================
