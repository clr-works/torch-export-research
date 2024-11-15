#!/bin/bash

# Default flags
CLEAN=false
BUILD=false

# Parse command-line arguments for cleaning and building flags
while getopts "cb" opt; do
  case $opt in
    c)
      CLEAN=true
      ;;
    b)
      CLEAN=true
      BUILD=true
      ;;
    *)
      echo "Usage: $0 [-c] to clean build directory, [-b] to clean and build"
      exit 1
      ;;
  esac
done

# Define the build directory
BUILD_DIR="./build"

# Clean the build directory if flag is set
if [ "$CLEAN" = true ]; then
  echo "Cleaning build directory..."
  rm -rf "$BUILD_DIR"
  echo "Build directory cleaned."
fi

# Only proceed with build if:
# 1. -b flag is set, OR
# 2. no flags were provided (default incremental build)
if [ "$BUILD" = true ] || [ $OPTIND -eq 1 ]; then
  echo "Creating build directory..."
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR" || exit 1

  # Set the Torch_DIR for CMake to find PyTorch
  TORCH_DIR="/home/kmh2266/hpml/fp/aoti_env/lib/python3.8/site-packages/torch/share/cmake/Torch"

  echo "Running cmake..."
  cmake -DTorch_DIR="$TORCH_DIR" ..

  echo "Building project..."
  make
  echo "Build complete!"
fi