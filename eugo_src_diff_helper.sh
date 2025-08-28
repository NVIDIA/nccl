#!/bin/bash

# Bash script to find differences in source files between fork and upstream NVIDIA/nccl repo
# This script helps identify newly added source files that need to be integrated into CMakeLists.txt files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
FORK_DIR="$(pwd)"
UPSTREAM_REPO="https://github.com/NVIDIA/nccl.git"
TEMP_DIR="/tmp/nccl-upstream-$$"
DIFF_OUTPUT_FILE="source_files_diff.txt"

echo -e "${BLUE}=== NCCL Source Files Diff Helper ===${NC}"
echo "Fork directory: $FORK_DIR"
echo "Upstream repo: $UPSTREAM_REPO"
echo "Temp directory: $TEMP_DIR"
echo

# Create temporary directory and clone upstream repo
echo -e "${YELLOW}Cloning upstream repository...${NC}"
git clone --depth 1 "$UPSTREAM_REPO" "$TEMP_DIR"

# Function to find source files in a directory
find_source_files() {
    local dir="$1"
    local base_dir="$2"
    
    if [ ! -d "$dir" ]; then
        return 0
    fi
    
    find "$dir" -type f \( -name "*.cc" -o -name "*.cu" -o -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | \
    sed "s|^$base_dir/||" | sort
}

# Function to compare directories and show differences
compare_directory() {
    local subdir="$1"
    local cmake_target="$2"
    
    echo -e "${GREEN}=== Comparing $subdir (maps to $cmake_target) ===${NC}"
    
    local fork_src_dir="$FORK_DIR/src/$subdir"
    local upstream_src_dir="$TEMP_DIR/src/$subdir"
    
    # Handle special case for root directory
    if [ "$subdir" = "." ]; then
        fork_src_dir="$FORK_DIR/src"
        upstream_src_dir="$TEMP_DIR/src"
    fi
    
    local fork_files_temp=$(mktemp)
    local upstream_files_temp=$(mktemp)
    
    # Get source files from both directories
    find_source_files "$fork_src_dir" "$FORK_DIR/src" > "$fork_files_temp"
    find_source_files "$upstream_src_dir" "$TEMP_DIR/src" > "$upstream_files_temp"
    
    # Filter files to only include those in the specific subdirectory
    if [ "$subdir" != "." ]; then
        grep "^$subdir/" "$fork_files_temp" > "${fork_files_temp}.filtered" 2>/dev/null || touch "${fork_files_temp}.filtered"
        grep "^$subdir/" "$upstream_files_temp" > "${upstream_files_temp}.filtered" 2>/dev/null || touch "${upstream_files_temp}.filtered"
        mv "${fork_files_temp}.filtered" "$fork_files_temp"
        mv "${upstream_files_temp}.filtered" "$upstream_files_temp"
    else
        # For root directory, exclude subdirectories that are handled separately
        grep -v -E "^(misc|plugin|transport|graph|register|ras|device)/" "$fork_files_temp" > "${fork_files_temp}.filtered" 2>/dev/null || touch "${fork_files_temp}.filtered"
        grep -v -E "^(misc|plugin|transport|graph|register|ras|device)/" "$upstream_files_temp" > "${upstream_files_temp}.filtered" 2>/dev/null || touch "${upstream_files_temp}.filtered"
        mv "${fork_files_temp}.filtered" "$fork_files_temp"
        mv "${upstream_files_temp}.filtered" "$upstream_files_temp"
    fi
    
    # Find files only in upstream (newly added)
    local new_files=$(comm -13 "$fork_files_temp" "$upstream_files_temp")
    
    # Find files only in fork (removed from upstream)
    local removed_files=$(comm -23 "$fork_files_temp" "$upstream_files_temp")
    
    if [ -n "$new_files" ]; then
        echo -e "${YELLOW}Files added in upstream (need to add to $cmake_target):${NC}"
        echo "$new_files" | sed 's/^/  /'
        echo
    fi
    
    if [ -n "$removed_files" ]; then
        echo -e "${RED}Files removed from upstream (may need to remove from $cmake_target):${NC}"
        echo "$removed_files" | sed 's/^/  /'
        echo
    fi
    
    if [ -z "$new_files" ] && [ -z "$removed_files" ]; then
        echo -e "${GREEN}No differences found${NC}"
        echo
    fi
    
    # Cleanup temp files
    rm -f "$fork_files_temp" "$upstream_files_temp"
}

# Create output file
echo "=== NCCL Source Files Diff Report ===" > "$DIFF_OUTPUT_FILE"
echo "Generated on: $(date)" >> "$DIFF_OUTPUT_FILE"
echo "Fork: $FORK_DIR" >> "$DIFF_OUTPUT_FILE"
echo "Upstream: $UPSTREAM_REPO" >> "$DIFF_OUTPUT_FILE"
echo >> "$DIFF_OUTPUT_FILE"

# Compare each directory mapping
echo -e "${BLUE}Starting directory comparisons...${NC}"
echo

# Directory mappings from the user's requirements
compare_directory "." "./src/CMakeLists.txt"
compare_directory "misc" "./src/CMakeLists.txt"
compare_directory "plugin" "./src/CMakeLists.txt"
compare_directory "transport" "./src/CMakeLists.txt"
compare_directory "graph" "./src/CMakeLists.txt"
compare_directory "register" "./src/CMakeLists.txt"
compare_directory "ras" "./src/ras/CMakeLists.txt"
compare_directory "device" "./src/device/CMakeLists.txt"

# Cleanup
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
rm -rf "$TEMP_DIR"

echo -e "${GREEN}=== Diff analysis complete! ===${NC}"
echo "Results have been displayed above and saved to: $DIFF_OUTPUT_FILE"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review the differences shown above"
echo "2. Add newly found source files to the appropriate CMakeLists.txt files"
echo "3. Update the TODO comment as requested"
echo "4. Test the build to ensure everything works correctly"
