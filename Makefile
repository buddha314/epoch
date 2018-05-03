include local.mk
CC=chpl
INCLUDES=-I$(BLAS_HOME)/include -I$(POSTGRES_HOME)
LIBS=-L${BLAS_HOME}/lib -lblas
SRCDIR=src
BINDIR=target
MODULES=-M$(CDO_HOME)/src -M$(NUMSUCH_HOME)/src -M$(CHARCOAL_HOME)/src -M$(CHINGON_HOME)/src
EXEC=epoch
DB_CREDS=../db_creds.txt
TESTDIR=test
default: all

all: $(SRCDIR)/Epoch.chpl
	$(CC) $(INCLUDES) $(LIBS) $(MODULES) -o $(BINDIR)/$(EXEC) $<

run:
	./$(BINDIR)/$(EXEC)

run-test: $(TESTDIR)/EpochTest.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(TESTDIR)/test $<; \
	./$(TESTDIR)/test;  \
	rm $(TESTDIR)/test
