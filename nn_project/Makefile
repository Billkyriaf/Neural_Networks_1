CC = g++

BUILD_DIR := ./make-build-debug-g
SRC_DIRS := ./src
INC_DIRS := ./include

# Colors
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m
BOLD = \033[1m

# Directories
LIBRARIES_SRC := $(shell find $(INC_DIRS) -name '*.cpp')
LIBRARIES_SRC := $(LIBRARIES_SRC:%=$(BUILD_DIR)/%.o)

NN_SRC := $(shell find $(SRC_DIRS) -name '*.cpp')
NN_SRC := $(NN_SRC:%=$(BUILD_DIR)/%.o)

# Every folder in ./src and ./include will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(shell find $(INC_DIRS) -type d)

# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CC_FLAGS := $(INC_FLAGS) -O3 -std=c++14

$(BUILD_DIR)/nn.out: $(NN_SRC) $(LIBRARIES_SRC)
	@echo
	@echo -e "        $(BOLD)Linking...$(NC)"
	@echo
	@$(CC) $(CC_FLAGS) -o $(BUILD_DIR)/nn.out $(NN_SRC) $(LIBRARIES_SRC)
	@echo -e "    $(GREEN)Build finished successfully!$(NC)"
	@echo

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	@echo -e "        $(BOLD)Compiling:$(NC) $(<)..."
	@$(CC) $(CC_FLAGS) -c $< -o $@

%.cpp:

build_nn: $(BUILD_DIR)/nn.out
run_nn: $(BUILD_DIR)/nn.out
	@echo
	@echo
	$(BUILD_DIR)/nn.out
	@echo
	@echo

.PHONY: clean
clean:
	@echo -e "$(RED)Clearing build directories...$(NC)"
	@rm -rf $(BUILD_DIR)/*
	@echo -e "$(GREEN)Done!$(NC)"