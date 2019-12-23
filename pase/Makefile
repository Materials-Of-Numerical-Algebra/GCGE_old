###########################################################

GCGEHOME = ..
PASEHOME = .
include $(GCGEHOME)/config/make.inc
.PHONY: all libs clean help test-csr test-hypre test-petsc test-phg test-slepc

###########################################################

all: 	help

install_libs = libgcge_pase

libs: $(install_libs)
	@echo "======================================="
	@echo "Build $(strip $(subst -, , $(install_libs))) complete."
	@echo "======================================="
	
libgcge_pase:
	@echo "======================================="
	@echo "        Making library GCGE APP        "
	@cd $(PASEHOME)/src;      $(MAKE) lib

test-csr: libs
	@cd $(PASEHOME)/example;  $(MAKE) run-csr-solver

test-slepc: libs
	@cd $(PASEHOME)/example;  $(MAKE) run-slepc-solver

clean:
	@cd $(PASEHOME)/src;      make clean; rm -rf *~
	@cd $(PASEHOME)/example;  make clean; rm -rf *~

cleanall: clean
	@cd $(PASEHOME)/lib;      $(RM) $(RMFLAGS) *.a
	@cd $(PASEHOME)/include;  $(RM) $(RMFLAGS) *.h

help:
	@echo " "
	@echo "make {libs|clean|cleanlibs|help}" 
	@echo " "
	@echo "   libs         - create all libraries"
	@echo " "
	@echo "   test-csr     - test library GCGE"
	@echo "   test-hypre   - test library GCGE HYPRE"
	@echo "   test-petsc   - test library GCGE PETSC"
	@echo "   test-phg     - test library GCGE PHG"
	@echo "   test-slepc   - test library GCGE SLPEC"
	@echo " "
	@echo "   clean        - remove temporary files except libraries"
	@echo "   cleanall     - remove libraries"
	@echo " "
	@echo "   help         - print this information"
	@echo " "
