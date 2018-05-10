include ../local.mk
CC=chpl
INCLUDES=-I$(BLAS_HOME)/include -I$(POSTGRES_HOME)
LIBS=-L${BLAS_HOME}/lib -lblas
SRCDIR=../src
CONFIG=examples.cfg
BINDIR=target
MODULES=-M$(CDO_HOME)/src -M$(NUMSUCH_HOME)/src -M$(CHARCOAL_HOME)/src -M$(CHINGON_HOME)/src
EXEC=epoch
DB_CREDS=../db_creds.txt
APPROX_DIR=approximators
default: xor sine

xor: $(APPROX_DIR)/XOR.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f xor.cfg;  \
	rm $(APPROX_DIR)/example

sine: $(APPROX_DIR)/Sine.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f sine.cfg;  \
	rm $(APPROX_DIR)/example
