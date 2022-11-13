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
LIBRARIES_SRC := $(shell find $(SRC_DIRS)/utils -name '*.cpp')
LIBRARIES_SRC += $(shell find $(SRC_DIRS)/mnist -name '*.cpp')
LIBRARIES_SRC := $(LIBRARIES_SRC:%=$(BUILD_DIR)/%.o)

KNN_SRC := $(shell find $(SRC_DIRS)/knn -name '*.cpp')
KNN_SRC += $(shell find $(SRC_DIRS) -name 'KNN_main.cpp')
KNN_SRC := $(KNN_SRC:%=$(BUILD_DIR)/%.o)

NCC_SRC := $(shell find $(SRC_DIRS)/ncc -name '*.cpp')
NCC_SRC += $(shell find $(SRC_DIRS) -name 'NCC_main.cpp')
NCC_SRC := $(NCC_SRC:%=$(BUILD_DIR)/%.o)

NCC_CLUSTER_SRC := $(shell find $(SRC_DIRS)/ncc_cluster -name '*.cpp')
NCC_CLUSTER_SRC += $(shell find $(SRC_DIRS) -name 'NCC_Cluster_main.cpp')
NCC_CLUSTER_SRC := $(NCC_CLUSTER_SRC:%=$(BUILD_DIR)/%.o)

# Every folder in ./src and ./include will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(shell find $(INC_DIRS) -type d)

# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CC_FLAGS := $(INC_FLAGS) -O3 -std=c++14

$(BUILD_DIR)/knn.out: $(KNN_SRC) $(LIBRARIES_SRC)
	@echo
	@echo -e "        $(BOLD)Linking...$(NC)"
	@echo
	@$(CC) $(CC_FLAGS) -o $(BUILD_DIR)/knn.out $(KNN_SRC) $(LIBRARIES_SRC)
	@echo -e "    $(GREEN)Build finished successfully!$(NC)"
	@echo

$(BUILD_DIR)/ncc.out: $(NCC_SRC) $(LIBRARIES_SRC)
	@echo
	@echo -e "        $(BOLD)Linking...$(NC)"
	@echo
	@$(CC) $(CC_FLAGS) -o $(BUILD_DIR)/ncc.out $(NCC_SRC) $(LIBRARIES_SRC)
	@echo -e "    $(GREEN)Build finished successfully!$(NC)"
	@echo

$(BUILD_DIR)/ncc_cluster.out: $(NCC_CLUSTER_SRC) $(LIBRARIES_SRC)
	@echo
	@echo -e "        $(BOLD)Linking...$(NC)"
	@echo
	@$(CC) $(CC_FLAGS) -o $(BUILD_DIR)/ncc_cluster.out $(NCC_CLUSTER_SRC) $(LIBRARIES_SRC)
	@echo -e "    $(GREEN)Build finished successfully!$(NC)"
	@echo

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	@echo -e "        $(BOLD)Compiling:$(NC) $(<)..."
	@$(CC) $(CC_FLAGS) -c $< -o $@

%.cpp:

build_knn: $(BUILD_DIR)/knn.out
run_knn: $(BUILD_DIR)/knn.out
	@mkdir -p images
	@echo
	@echo
	$(BUILD_DIR)/knn.out -d ./data -k 7 -t 16 -n 10000 -s 0
	@echo
	@echo


build_ncc: $(BUILD_DIR)/ncc.out
run_ncc: $(BUILD_DIR)/ncc.out
	@mkdir -p images
	@echo
	@echo
	$(BUILD_DIR)/ncc.out -d ./data -n 10000 -s 0
	@echo
	@echo

build_ncc_cluster: $(BUILD_DIR)/ncc_cluster.out
run_ncc_cluster: $(BUILD_DIR)/ncc_cluster.out
	@mkdir -p images
	@mkdir -p pre_fit
	@echo
	@echo
	$(BUILD_DIR)/ncc_cluster.out -d ./data -c 350 -fit
	@echo
	@echo

.PHONY: clean
clean:
	@echo -e "$(RED)Clearing build directories...$(NC)"
	@rm -rf $(BUILD_DIR)/*
	@rm -rf images/*
	@echo -e "$(GREEN)Done!$(NC)"