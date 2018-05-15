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
CLASSY_DIR=classifiers
default: linear poly sine xor binon detect3

xor: $(APPROX_DIR)/XOR.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f xor.cfg;  \
	rm $(APPROX_DIR)/example

sine: $(APPROX_DIR)/Sine.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f sine.cfg;  \
	rm $(APPROX_DIR)/example

linear: $(APPROX_DIR)/Linear.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f linear.cfg;  \
	rm $(APPROX_DIR)/example

binon: $(APPROX_DIR)/BiNon.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f binon.cfg;  \
	rm $(APPROX_DIR)/example

detect3: $(CLASSY_DIR)/Detect3.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(CLASSY_DIR)/example $<; \
	./$(CLASSY_DIR)/example -f detect3.cfg;  \
	rm $(CLASSY_DIR)/example

poly: $(APPROX_DIR)/Polynomial.chpl
	$(CC) -M$(SRCDIR) $(MODULES) $(FLAGS) ${INCLUDES} ${LIBS} -o $(APPROX_DIR)/example $<; \
	./$(APPROX_DIR)/example -f poly.cfg;  \
	rm $(APPROX_DIR)/example
