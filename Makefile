CMAKE ?= cmake
BUILD_DIR ?= build
TARGET ?= ocr_engine
BUILD_TYPE ?= Debug

.PHONY: all configure build run clean rebuild clean-run release-configure release-build release-run

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j

run: build
	./$(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

clean-run: clean run

release-configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

release-build: release-configure
	$(CMAKE) --build $(BUILD_DIR) -j

release-run: release-build
	./$(BUILD_DIR)/$(TARGET)
