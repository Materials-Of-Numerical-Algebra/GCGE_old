############################################################
GCGEHOME = ../..
include $(GCGEHOME)/config/make.inc

############################################################
OBJS1  = test_ops.o

EXE1   = test_ops.exe 

INC    = -I$(GCGEINC) -I$(GCGEPASEINC)
LIBS   = $(LIBGCGEPASE) $(LIBGCGECSR) $(LIBGCGE) $(LIBLAPACK) $(LIBBLAS) $(LIBF)

############################################################
MAT_A = $(GCGEHOME)/test/data/testA
MAT_M = $(GCGEHOME)/test/data/testB
PARA = -file_A $(MAT_A) -file_B $(MAT_M)

############################################################

all:	help

exe:    $(EXE1)

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

############################################################
