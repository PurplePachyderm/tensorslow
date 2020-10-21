# Vars

CC=g++

CPPFLAGS=-Wall
OPT_FLAGS=-Ofast
TEST_FLAGS=-g

SRC=src
BIN=bin
LIB=lib
TEST=test
PERF=perf
EX=examples

OBJ_FILES=$(patsubst $(SRC)/%.cpp, $(LIB)/%.o, $(wildcard $(SRC)/*.cpp))
TEST_FILES=$(patsubst $(TEST)/%.cpp, $(BIN)/%_test, $(wildcard $(TEST)/*.cpp))
PERF_FILES=$(patsubst $(PERF)/%.cpp, $(BIN)/%_perf, $(wildcard $(PERF)/*.cpp))
EX_FILES=$(patsubst $(EX)/%.cpp, $(BIN)/%_example, $(wildcard $(EX)/*.cpp))

SO_NAME=$(LIB)/tensorflow.so



# Targets

all: lib test perf examples


lib: $(SO_NAME)
$(SO_NAME): $(OBJ_FILES)
	$(CC) $(CPPFLAGS) $(OPT_FLAGS) -shared -o $@ $^
$(OBJ_FILES): $(LIB)/%.o : $(SRC)/%.cpp
	$(CC) $(CPPFLAGS) $(OPT_FLAGS) $< -c -o $@


test: $(SO_NAME) $(TEST_FILES)
$(TEST_FILES): $(BIN)/%_test: $(TEST)/%.cpp
	$(CC) $< $(CPPFLAGS) $(OPT_FLAGS) $(TEST_FLAGS) -o $@


perf:  $(SO_NAME) $(PERF_FILES)
$(PERF_FILES): $(BIN)/%_perf: $(PERF)/%.cpp
	$(CC) $< $(CPPFLAGS) $(OPT_FLAGS) -o $@


examples:  $(SO_NAME) $(EX_FILES)
$(EX_FILES): $(BIN)/%_example: $(EX)/%.cpp
	$(CC) $< $(CPPFLAGS) $(OPT_FLAGS) -o $@



# Phonies

.PHONY: clean clean_lib clean_o clean_so clean_bin clean_test clean_perf clean_examples


clean: clean_lib clean_test clean_perf clean_examples

clean_lib:
	find $(LIB) -name "*.o" -type f -delete
	find $(LIB) -name "*.so" -type f -delete

clean_o:
	find $(LIB) -name "*.o" -type f -delete

clean_so:
	find $(LIB) -name "*.so" -type f -delete

clean_bin:
	find $(BIN) -name "*_test" -type f -delete
	find $(BIN) -name "*_perf" -type f -delete
	find $(BIN) -name "*_example" -type f -delete

clean_test:
	find $(BIN) -name "*_test" -type f -delete

clean_perf:
	find $(BIN) -name "*_perf" -type f -delete

clean_examples:
	find $(BIN) -name "*_example" -type f -delete
