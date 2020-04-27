########################################################################
GCGEHOME = ../..
PASEHOME = ..
include $(GCGEHOME)/config/make.inc
.PHONY: all lib clean help

########################################################################


INC         = -I$(GCGEINC) -I$(PASEHOME)/include $(HYPREINC) $(PETSCINC) $(PHGINC) $(SLEPCINC)
LIBGCGEPASE = $(PASEHOME)/lib/libgcge_pase.a

########################################################################
all:    help

exe:

clean:
	@$(RM) $(RMFLAGS) *.o *.exe

help:
	@echo "make {lib|clean|help}"
	@echo " "
	@echo "   exe   - compile all executables"
	@echo " "
	@echo "   xx    - test xx"
	@echo " "
	@echo "   clean - remove temporary files"
	@echo "   help  - print this message"
	@echo " "

run-convert:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_convert.exe

run-convert_phg:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_convert_phg.exe
	
run-mg:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_mg.exe

run-mg_phg:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_mg_phg.exe

run-mg_petsc:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_mg_petsc.exe

run-pase:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_pase.exe

run-amg:
	$(DEBUGEXEC) $(DEBUGFLAGS) ./test_amg.exe

########################################################################

test_convert:
	$(CC)  $(CFLAGS) $(HYPREFLAGS) $(PETSCFLAGS) -c test_convert.c     -o test_convert.o     $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_convert.exe ./test_convert.o $(LIBGCGEPASE) $(LIBPHG) $(LIBPETSC) $(LIBHYPRE) 

test_convert_phg:
	$(CC)  $(CFLAGS) $(HYPREFLAGS) $(PHGFLAGS)   -c test_convert_phg.c -o test_convert_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_convert_phg.exe ./test_convert_phg.o $(LIBGCGEPASE) $(LIBPHG) $(LIBPETSC) $(LIBHYPRE) 

test_mg:
	$(CC)  $(CFLAGS) -c test_mg.c -o test_mg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg.exe ./test_mg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_petsc:
	$(CC)  $(CFLAGS) $(PETSCFLAGS) -c test_mg_petsc.c -o test_mg_petsc.o $(PETSCINC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_petsc.exe ./test_mg_petsc.o $(LIBPETSC) 

test_mg_phg:
	$(CC)  $(CFLAGS) -c test_mg_phg.c -o test_mg_phg.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg.exe ./test_mg_phg.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_phg_slepc:
	$(CC)  $(CFLAGS) -c test_phg_slepc.c -o test_phg_slepc.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_phg_slepc.exe ./test_phg_slepc.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 


test_mg_phg_petsc:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc.c -o test_mg_phg_petsc.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_100.exe ./test_mg_phg_petsc.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 
	#$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc.exe ./test_mg_phg_petsc.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_12:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_12.c -o test_mg_phg_petsc_12.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_12.exe ./test_mg_phg_petsc_12.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev25:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev25.c -o test_mg_phg_petsc_nev25.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev25_1e12.exe ./test_mg_phg_petsc_nev25.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev50:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev50.c -o test_mg_phg_petsc_nev50.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev50_1e12.exe ./test_mg_phg_petsc_nev50.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev100:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev100.c -o test_mg_phg_petsc_nev100.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev100_1e12.exe ./test_mg_phg_petsc_nev100.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev200:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev200.c -o test_mg_phg_petsc_nev200.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev200_1e12.exe ./test_mg_phg_petsc_nev200.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev400:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev400.c -o test_mg_phg_petsc_nev400.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev400_1e12.exe ./test_mg_phg_petsc_nev400.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev800:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev800.c -o test_mg_phg_petsc_nev800.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev800_1e12.exe ./test_mg_phg_petsc_nev800.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_nev1600:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_nev1600.c -o test_mg_phg_petsc_nev1600.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_nev1600_1e12.exe ./test_mg_phg_petsc_nev1600.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_mg_phg_petsc_ksp:
	$(CC)  $(CFLAGS) -c test_mg_phg_petsc_ksp.c -o test_mg_phg_petsc_ksp.o $(INC) 
	$(CC)  $(CFLAGS) -c get_mat_phg.c -o get_mat_phg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_mg_phg_petsc_ksp.exe ./test_mg_phg_petsc_ksp.o ./get_mat_phg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_pase:
	$(CC)  $(CFLAGS) -c test_pase.c -o test_pase.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_pase.exe ./test_pase.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

test_amg:
	$(CC)  $(CFLAGS) -c test_amg.c -o test_amg.o $(INC) 
	$(CLINKER)  $(LDFLAGS)  -o ./test_amg.exe ./test_amg.o $(LIBGCGEPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBPHG) $(LIBHYPRE) $(LIBGCGE) 

########################################################################
