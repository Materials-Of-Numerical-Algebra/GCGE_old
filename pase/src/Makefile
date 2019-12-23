########################################################################
GCGEHOME = ../..
PASEHOME = ..
include $(GCGEHOME)/config/make.inc
.PHONY: all lib clean help

########################################################################
FILES   = pase_convert.c gcge_app_pase.c pase_solver.c pase_matvec.c \
          pase_ops.c pase_mg.c pase_amg.c pase_param.c
HEADERS = pase_convert.h gcge_app_pase.h pase_solver.h pase_config.h \
          pase_matvec.h pase_mg.h pase_ops.h pase.h pase_param.h     \
          pase_amg.h
OBJS    = ${FILES:.c=.o}

INC     = -I$(GCGEINC) $(SLEPCINC) $(HYPREINC) $(PETSCINC) $(PHGINC)
LIB     = $(PASEHOME)/lib/libgcge_pase.a

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
	@echo 'Creating header files $(PASEHOME)/include/{$(HEADERS)}'
	@$(ARCH) $(ARCHFLAGS) $(LIB) $(OBJS)
	@$(RANLIB) $(LIB)
	@$(CP) -fR pase_*.h          $(PASEHOME)/include
	@$(CP) -fR pase.h            $(PASEHOME)/include
	@$(CP) -fR gcge_app_pase.h   $(PASEHOME)/include

$(OBJS): $(HEADERS)
########################################################################
