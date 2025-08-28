# LLM Prompt for NCCL CMake Fork Maintenance

## Context

This is a CMake-specific port of NVIDIA's NCCL library that exclusively leverages Clang for both host and device code compilation. The fork needs to be periodically synchronized with the upstream NVIDIA/nccl repository.

## Task Description

Using an LLM, write a bash script that finds the diff in source files for our fork and the upstream repo for the following directories:

### Directory Mappings

The following directories in the fork map to their corresponding CMakeLists.txt files:

a. `./` → maps to `./src/CMakeLists.txt`
b. `./misc` → maps to `./src/CMakeLists.txt`
c. `./plugin` → maps to `./src/CMakeLists.txt`
d. `./transport` → maps to `./src/CMakeLists.txt`
e. `./graph` → maps to `./src/CMakeLists.txt`
f. `./register` → maps to `./src/CMakeLists.txt`
g. `./ras` → maps to `./src/ras/CMakeLists.txt`
h. `./device` → maps to `./src/device/CMakeLists.txt`

## Requirements

1. **Diff Analysis**: The script should compare source files between the fork and upstream repository for each of the specified directories.

2. **File Detection**: Find newly added source files in upstream that need to be integrated into the corresponding CMakeLists.txt files.

3. **Output Format**: Provide clear output showing:

   - Files added in upstream (need to add to CMakeLists.txt)
   - Files removed from upstream (may need to remove from CMakeLists.txt)
   - Which CMakeLists.txt file each directory maps to

4. **File Types**: Focus on source files including: `.cc`, `.cu`, `.c`, `.cpp`, `.h`, `.hpp`

## Implementation Notes

- The script should clone the upstream repository temporarily for comparison
- Use standard Unix tools like `find`, `comm`, `sort` for file comparison
- Provide colored output for better readability
- Clean up temporary files after analysis
- Save results to a summary file

## Post-Processing Tasks

After running the diff script:

1. **Update CMakeLists.txt**: Add newly found source files to the appropriate CMakeLists.txt files in the correct sections
2. **Update Comments**: Replace the TODO comment `@TODO+:Slava_n_Ben:check either they are public or private. More likely, the latter.` with `NOTE: these are private headers`
3. **File Organization**: Ensure files are added in alphabetical order within their respective sections
4. **Unity Build**: Add new plugin files to the `SKIP_UNITY_BUILD_INCLUSION` properties if they follow the plugin pattern

## Expected Workflow

1. Run the bash script to identify differences
2. Review the output to understand what files were added/removed
3. Update the CMakeLists.txt files with new source files
4. Update any TODO comments as specified
5. Test the build to ensure everything compiles correctly

## Repository Information

- **Fork**: https://github.com/eugo-inc/nccl-cmake/tree/master
- **Upstream**: https://github.com/NVIDIA/nccl
- **Build System**: CMake with Clang compiler
- **Target**: CUDA device and host code compilation

This maintenance process ensures the fork stays synchronized with upstream changes while maintaining the CMake build system and Clang compatibility.
