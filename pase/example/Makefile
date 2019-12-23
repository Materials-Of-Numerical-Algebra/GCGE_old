############################################################
GCGEHOME = ../..
PASEHOME = ..
include $(GCGEHOME)/config/make.inc
.PHONY: all exe clean help
############################################################

EXE_CSR    = pase_csr_solver.exe
EXE_SLEPC  = pase_slepc_solver.exe
           
OBJ_CSR    = pase_csr_solver.o
OBJ_SLEPC  = pase_slepc_solver.o
           
C_CSR      = pase_csr_solver.c
C_SLEPC    = pase_slepc_solver.c
           
INC_CSR    = $(CSRFLAGS)   -I$(PASEHOME)/include -I$(GCGEINC)          
INC_SLEPC  = $(SLEPCFLAGS) -I$(PASEHOME)/include -I$(GCGEINC) $(SLEPCINC) $(PETSCINC)
          
LIBGCGEPASE     = $(PASEHOME)/lib/libgcge_pase.a
LIBS_CSR        = $(LIBGCGECSR)   $(LIBGCGE) $(LIBLAPACK) $(LIBBLAS) $(LIBF)
LIBS_SLEPC      = $(LIBGCGESLEPC) $(LIBGCGE) $(LIBSLEPC)

LIBS_PASE_CSR   = $(LIBGCGEPASE)  $(LIBS_CSR)   
LIBS_PASE_SLEPC = $(LIBGCGEPASE)  $(LIBS_SLEPC) 

############################################################

all:    help

exe:    $(EXE_CSR) $(EXE_SLEPC)

clean:
	@$(RM) $(RMFLAGS) *.o *.exe

help:
	@echo " "
	@echo "make {exe|run-xx-solver|clean|help}"
	@echo " "
	@echo "   exe              - compile all executables"
	@echo " "
	@echo "   run-csr-solver   - run gcge solver in csr"
	@echo "   run-slepc-solver - run gcge solver in slepc"
	@echo " "
	@echo "   clean            - remove temporary files"
	@echo "   help             - print this message"
	@echo " "

############################################################

run-csr-solver: $(EXE_CSR)
	$(DEBUGEXEC) $(DEBUGFLAGS) ./$(EXE_CSR)

run-slepc-solver: $(EXE_SLEPC)
	$(DEBUGEXEC) $(DEBUGFLAGS) $(EXEC) $(EXECFLAGS) ./$(EXE_SLEPC)

############################################################

$(EXE_CSR): $(OBJ_CSR)
	@echo ''
	@echo 'Creating executable $(EXE_CSR)'
	$(CLINKER) $(LDFLAGS) -o $@ $< $(LIBS_PASE_CSR)
	@echo ''

$(EXE_SLEPC): $(OBJ_SLEPC)
	@echo ''
	@echo 'Creating executable $(EXE_SLEPC)'
	$(CLINKER) $(LDFLAGS) -o $@ $< $(LIBS_PASE_SLEPC)
	@echo ''

############################################################

$(OBJ_CSR):$(C_CSR)
	@echo '$(CC) $(CFLAGS) -c $(C_CSR)'
	$(CC)  $(CFLAGS) -c $(C_CSR) -o $(OBJ_CSR) $(INC_CSR)

$(OBJ_SLEPC):$(C_SLEPC)
	@echo '$(CC) $(CFLAGS) -c $(C_SLEPC)'
	$(CC)  $(CFLAGS) -c $(C_SLEPC) -o $(OBJ_SLEPC) $(INC_SLEPC)

############################################################
