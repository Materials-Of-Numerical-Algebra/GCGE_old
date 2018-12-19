########################################################################
GCGEHOME = ..
include $(GCGEHOME)/config/make.inc
.PHONY: all lib clean help

########################################################################
FILES   = pase_convert.c pase_matvec.c pase_mg.c pase_solver.c pase_ops.c
HEADERS = pase_convert.h pase_config.h pase_matvec.h pase_mg.h pase_solver.h pase_ops.h pase.h
OBJS    = ${FILES:.c=.o}

INC     = -I$(GCGEINC) -I$(PETSCINC) -I$(HYPREINC)
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
#	@$(CP) -fR pase_*.h $(GCGEINC)
#	@$(CP) -fR pase.h   $(GCGEINC)

$(OBJS): $(HEADERS)
########################################################################

test:
	$(CC)  $(CFLAGS) -c ./test_convert.c -o ./test_convert.o $(INC)
	$(CLINKER)  $(LDFLAGS)  -o ./test_convert.exe ./test_convert.o $(LIBPASE) $(LIBPETSC) $(LIBHYPRE) 
