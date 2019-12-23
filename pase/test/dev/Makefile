############################################################
GCGEHOME = ../../..
PASEHOME = ../..
include $(GCGEHOME)/config/make.inc

############################################################
OBJS1  = test_csr_pase_solver.o
OBJS2  = test_ops.o
OBJS3  = test_pase_1.o
OBJS4  = test_slepc_pase_ops.o

EXE1   = ./test_csr_pase_solver.exe 
EXE2   = ./test_ops.exe 
EXE3   = ./test_pase_1.exe 
EXE4   = ./test_slepc_pase_ops.exe 

LIBGCGEPASE = $(PASEHOME)/lib/libgcge_pase.a

#INC    = -I$(GCGEINC)
#LIBS   = $(LIBPASE) $(LIBGCGECSR) $(LIBGCGE) $(LIBLAPACK) $(LIBBLAS) $(LIBF)

INC    = -I$(PASEHOME)/include -I$(GCGEINC) $(HYPREINC) $(PETSCINC) $(PHGINC) $(SLEPCINC) 
LIBS   = $(LIBGCGEPASE) $(LIBGCGECSR) $(LIBGCGESLEPC) $(LIBGCGE) $(LIBLAPACK) $(LIBBLAS) $(LIBF) $(LIBSLEPC)

############################################################
MAT_A = $(PASEHOME)/test/data/testA
MAT_M = $(PASEHOME)/test/data/testB
PARA = -file_A $(MAT_A) -file_B $(MAT_M)

############################################################

all:	help

exe1:    $(EXE1)

exe2:    $(EXE2)

exe3:    $(EXE3)

exe4:    $(EXE4)

clean:
	@$(RM) $(RMFLAGS) *.o *.exe

help:
	@echo " "
	@echo "make {exe|run-mesh-ani2D|clean|help}"
	@echo " "
	@echo "   exe   - compile two executables"
	@echo " "
	@echo "   xx    - test xx"
	@echo " "
	@echo "   clean - remove temporary files"
	@echo "   help  - print this message"
	@echo " "

run-ops:
	$(EXE1)
	
run-ops-debug:
	$(DEBUGEXEC) $(DEBUGFLAGS) $(EXE1)

run-debug:
	valgrind -v $(EXE1)

############################################################
.f.o:
	@echo '$(F77) -c $(FFLAGS) ' $*.f
	@$(F77) $(FFLAGS) -c $*.f -o $*.o $(INC)
.c.o:
	@echo '$(CC)  -c $(CFLAGS) ' $*.c
	$(CC)  $(CFLAGS) -c $*.c -o $*.o $(INC)
.cpp.o:
	@echo '$(CPP)  -c $(CPPFLAGS) ' $*.cpp
	@$(CPP)  $(CPPFLAGS) -c $*.cpp -o $*.o $(INC)

############################################################
$(EXE1): $(OBJS1)
	@echo ''
	@echo 'Creating executable bin/xx.exe'
	$(CLINKER) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo ''

$(EXE2): $(OBJS2)
	@echo ''
	@echo 'Creating executable bin/xx.exe'
	$(CLINKER) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo ''

$(EXE3): $(OBJS3)
	@echo ''
	@echo 'Creating executable bin/xx.exe'
	$(CLINKER) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo ''

$(EXE4): $(OBJS4)
	@echo ''
	@echo 'Creating executable bin/xx.exe'
	@$(CLINKER) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo ''

############################################################
