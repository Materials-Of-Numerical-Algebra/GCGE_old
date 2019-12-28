############################################################
INSTALLDIR   = ${HOME}/GCGE-3.0

############################################################
# DEFINE libs and dirs of include of OEM package
############################################################

LIBLAPACK    = -llapack
LIBBLAS      = -lblas

############################################################
# DEFINE libs and dirs of include of OEM package
############################################################

PETSCFLAGS   = -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  
PETSCINC     = -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBPETSC     = -L${PETSC_DIR}/${PETSC_ARCH}/lib -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/5 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/local/lib -lpetsc -llapack -lblas -lpthread -lm -lX11 -lstdc++ -ldl -lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl

SLEPCFLAGS   = -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  
SLEPCINC     = -I${SLEPC_DIR}/include -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBSLEPC     = -Wl,-rpath,${SLEPC_DIR}/${PETSC_ARCH}/lib -L${SLEPC_DIR}/${PETSC_ARCH}/lib -L${PETSC_DIR}/${PETSC_ARCH}/lib -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/5 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/local/lib -lslepc -lpetsc -llapack -lblas -lpthread -lm -lX11 -lstdc++ -ldl -lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl

HYRPREFLAGS  = -Wall -DHAVE_CONFIG_H -DHYPRE_TIMING
HYPREINC     = -I${HYPRE_DIR}/src/hypre/include
LIBHYPRE     = -Wall -L${HYPRE_DIR}/src/hypre/lib -lHYPRE -lm -lstdc++

PHGFLAGS     = -Wall -Wmissing-field-initializers -fopenmp -D__PHG__ 
PHGINC       = -I${PHG_DIR}/include
LIBPHG       = -L${PHG_DIR}/src -fopenmp -Wl,-rpath,${PHG_DIR}/src -lphg  -lmpfr  -lgmp  -lumfpack -lamd  -llapack -lblas  -lm  -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/5 -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../../lib -L/lib/x86_64-linux-gnu -L/lib/../lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/5/../../.. -lgfortran -lm -lmpifort -lmpi -lquadmath  -lgfortran -lm -lmpifort -lmpi -lquadmath -Wl,-rpath,/usr/local/lib

############################################################
VALGRIND      = valgrind
VALGRINDFLAGS = --tool=memcheck --leak-check=full -v --track-origins=yes --show-reachable=yes --read-var-info=yes
############################################################
# DEFINE compilers, linkers and execution
############################################################
F77          = mpif77
CC           = mpicc
CPP          = mpicc
EXEC         = mpiexec
EXECFLAGS    = -n 2

FLINKER      = $(F77)
CLINKER      = $(CC)
CPPLINKER    = $(CPP)

DEBUG        = -g
FFLAGS       = -O2 -fPIC $(DEBUG)
CFLAGS       = -O2 -fPIC $(DEBUG)
CPPFLAGS     = -O2 -fPIC $(DEBUG)

DEBUGEXEC    = $(VALGRIND)
DEBUGFLAGS   = $(VALGRINDFLAGS)

############################################################
# DEFINE for mixed code
############################################################
# linking Fortran
LIBF         = -lgfortran -lm
# linking CPP
LIBCPP       = -lstdc++
# other exist libs
LIBOE        =
# linker flags
LDFLAGS      =

############################################################
# DEFINE dirs
############################################################

GCGEBIN       = $(GCGEHOME)/bin
GCGEINC       = $(GCGEHOME)/include
GCGELIB       = $(GCGEHOME)/lib
GCGESRC       = $(GCGEHOME)/src

############################################################
version       = 3.0

############################################################
# DEFINE libs and dirs of include
############################################################

LIBGCGE       = $(GCGELIB)/libgcge_core-$(version).a

LIBGCGECSR    = $(GCGELIB)/libgcge_csr-$(version).a
LIBGCGEHYPRE  = $(GCGELIB)/libgcge_hypre-$(version).a
LIBGCGEPHG    = $(GCGELIB)/libgcge_phg-$(version).a
LIBGCGEPETSC  = $(GCGELIB)/libgcge_petsc-$(version).a
LIBGCGESLEPC  = $(GCGELIB)/libgcge_slepc-$(version).a

############################################################

MAKE         = make
CP           = /bin/cp
RM           = /bin/rm
RMFLAGS      = -f
ARCH         = ar
ARCHFLAGS    = cr
RANLIB       = ranlib
############################################################
