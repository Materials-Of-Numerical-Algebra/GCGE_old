########################################################################
GCGEHOME = ..
include $(GCGEHOME)/config/make.inc
.PHONY: all lib clean help

########################################################################
FILES   = pase_convert.c gcge_app_pase.c pase_solver.c pase_matvec.c \
          pase_ops.c pase_mg.c pase_amg.c pase_param.c
HEADERS = pase_convert.h gcge_app_pase.h pase_solver.h pase_config.h \
          pase_matvec.h pase_mg.h pase_ops.h pase.h pase_param.h     \
          pase_amg.h
OBJS    = ${FILES:.c=.o}

INC     = -I$(GCGEINC) -I$(SLEPCINC) -I$(HYPREINC) -I$(PETSCINC) -I$(PHGINC)
LIB     = $(LIBPASE)

########################################################################
all:    help

lib:    $(LIB)

clean:
	@$(RM) $(RMFLAGS) *.o

help:
	@echo "make {lib|clean|help}"
	@echo " "
	@echo "   lib   - create library $(LIB)"
	@echo " "
	@echo "   clean - remove temporary files"
	@echo "   help  - print this message"
	@echo " "

########################################################################
.f.o:
	@echo '$(F77) -c $(FFLAGS) ' $*.f
	@$(F77) $(FFLAGS) -c $*.f -o $*.o $(INC)
.c.o:
	@echo '$(CC)  -c $(CFLAGS) ' $*.c
	$(CC)  $(CFLAGS) -c $*.c -o $*.o $(INC)
.cpp.o:
	@echo '$(CPP)  -c $(CPPFLAGS) ' $*.cpp
	@$(CPP)  $(CPPFLAGS) -c $*.cpp -o $*.o $(INC)

########################################################################
$(LIB): $(OBJS)
	@echo ''
	@echo 'Creating library $(LIB)'
	@echo 'Creating header files $(GCGEINC)/{$(HEADERS)}'
	@$(ARCH) $(ARCHFLAGS) $(LIB) $(OBJS)
	@$(RANLIB) $(LIB)
	@$(CP) -fR $(HEADERS) $(GCGEINC)
#	@$(CP) -fR pase_*.h $(GCGEINC)
#	@$(CP) -fR pase.h   $(GCGEINC)

$(OBJS): $(HEADERS)
########################################################################

test:
	$(CC)  $(CFLAGS) -c ./test_convert.c -o ./test_convert.o $(INC)
	$(CLINKER)  $(LDFLAGS)  -o ./test_convert.exe ./test_convert.o $(LIBPASE) $(LIBPETSC) $(LIBHYPRE) 

test_phg:
	$(CC)  $(CFLAGS) -c ./test_convert_phg.c -o ./test_convert_phg.o $(INC)
	$(CLINKER)  $(LDFLAGS)  -o ./test_convert_phg.exe ./test_convert_phg.o $(LIBPASE) $(LIBPHG) $(LIBPETSC) $(LIBHYPRE) 

test_mg:
	@$(CC)  $(CFLAGS) -c ./test_mg.c -o ./test_mg.o $(INC)
	@$(CLINKER)  $(LDFLAGS)  -o ./test_mg.exe ./test_mg.o $(LIBPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBHYPRE) $(LIBGCGE) 

test_pase:
	@$(CC)  $(CFLAGS) -c ./test_pase.c -o ./test_pase.o $(INC)
	@$(CLINKER)  $(LDFLAGS)  -o ./test_pase.exe ./test_pase.o $(LIBPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBHYPRE) $(LIBGCGE) 

test_amg:
	@$(CC)  $(CFLAGS) -c ./test_amg.c -o ./test_amg.o $(INC)
	@$(CLINKER)  $(LDFLAGS)  -o ./test_amg.exe ./test_amg.o $(LIBPASE) $(LIBGCGESLEPC) $(LIBSLEPC) $(LIBHYPRE) $(LIBGCGE) 
