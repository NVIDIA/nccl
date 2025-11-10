# Contributing to NCCL

Thank you for your interest in contributing to NCCL! We appreciate the time and effort you're putting into helping improve the library. This document will help guide you through the contribution process.

## Before You Start

To help ensure your contribution can be accepted smoothly:

- **Open an issue first** for significant changes or new features. This allows us to discuss the approach before you invest significant time.
- **Check existing issues** to see if someone else is already working on something similar.
- **Ask questions** if you're unsure about anything. We're happy to help!

## Getting Started

1. **Fork the Repository**: Fork [https://github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl) and clone your fork locally.

2. **Create a Branch**: Create a feature branch for your work.
   ```bash
   git checkout -b <my-feature-branch>
   ```

3. **Commit your Changes**: Commit and push into feature branch
   ```bash
   git add <files-to-commit>
   git commit --message <descriptive message, see below>
   git push origin <my-feature-branch> --set-upstream
   ```
## What We're Looking For

We welcome contributions in the following areas:

- **Bug fixes**: Reproducible bugs with clear fixes are always appreciated
- **Performance improvements**: Targeted optimizations, guarded by appropriate checks
- **Platform support**: Extending NCCL to new platforms, network fabrics, or PCI bridges
- **Documentation**: Improvements to README, comments, or usage examples

## Making Your Contribution Successful

To help us review and integrate your contribution efficiently:

### Scope and Focus
- Keep changes focused on a single issue or feature
- Break large changes into smaller, reviewable pieces when possible
- Avoid mixing unrelated changes (e.g., bug fix + formatting changes) in one PR

### Quality Standards
- Ensure your code compiles without warnings
- Test thoroughly on relevant hardware configurations
- Performance claims should be backed by measurements
- Public API changes require discussion (see below)

### Common Challenges

Some types of contributions require extra discussion before we can accept them:

- **Public API changes**: NCCL's API stability is important to users. If your contribution changes public APIs, please open an issue to discuss the design first. We need to ensure backward compatibility and consistency.

- **Architecture-specific code**: NCCL supports specific GPU architectures and network configurations. Contributions targeting unsupported architectures may not be accepted unless there's a clear plan for ongoing maintenance.

- **Workarounds for external issues**: If a change works around a problem in another component (drivers, libraries, frameworks), we may need to address it differently. Let's discuss the root cause first.

- **Incomplete implementations**: Partial features or fixes that don't fully address the problem are difficult to maintain. If you need help completing an implementation, let us know - we're happy to assist!

## New Features

If you're proposing a substantial new feature (e.g., new collective operations, transport mechanisms, algorithms, or significant architectural changes), we follow a more structured process:

1. **Open an issue** describing the feature and use case
2. **Discussion phase**: We'll discuss whether it fits NCCL's direction
3. **Document Design**: Document the proposed architecture, a testing and validation plan as well as any known performance impact.
4. **Review**: The design will be reviewed by NCCL maintainers
5. **Implementation**: We recommend that you wait with the implementation until the review has concluded. Once there's a mutual agreement on the approach, proceed with implementation
6. **Pull request**: Submit your PR referencing the original issue and design doc


## Code Style

NCCL follows these coding conventions:

### General Guidelines
- Use 2 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Follow K&R brace style for C/CUDA code
- Use clear, descriptive variable names

### Naming Conventions
- Functions and variables: `ncclCamelCase`
- Macros and constants: `UPPER_CASE_WITH_UNDERSCORES`
- Struct/type names: `ncclFooBar`

### CUDA-Specific
- Minimize register usage in performance-critical kernels
- Avoid warp divergence where possible
- Document kernel launch configurations and occupancy considerations
- Prefer `cudaLaunchKernel` over `<<<>>>` syntax

### Return Codes
- NCCL functions should return a `ncclResult_t`
- All function calls should be guarded by `NCCLCHECK` or similar macro
- All external function calls should be guarded with `CUDACHECK`, `SYSCHECK`, `PTHREADCHECK`, etc.

### Memory Management
- Always allocate memory through NCCL alloc functions, e.g. `ncclCalloc()`
- Use appropriate memory fences for synchronization
- Free resources in reverse order of allocation

### Comments and Documentation
- Write clear comments for complex algorithms
- Document assumptions and invariants
- Explain "why" not just "what" for non-obvious code

### Code Formatting
- Avoid trailing white-spaces
- Try to match the existing style in files you're modifying

## Pull Request Guidelines

### Before Submitting
1. Rebase your branch on the latest master branch
2. Run `make` to ensure clean build with no warnings

### Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- First line: brief summary (50 chars or less)
- Second line blank
- Detailed explanation including the Problem, Solution, and any Limitations
- Reference issue numbers: "Fixes #123"

An example commit message is:
```text
Comment          | Commit message
-----------------|--------------------------------------------------------
Title            | Fix crash in proxy on systems with more than 2 GPUs
Blank line       | 
Problem          | When a system has more than 2 GPUs, the table we use to
                 | store addresses overflows.
Solution         | This fix increases the size of the table to the maximum
                 | number of GPUs we can have within a node.
Limitations and  | We may want to make the table size dynamic to save
Caveats          | memory, but since it is a part of a struct, it is easier
                 | for now to keep the code simple.
```

### Signed Commits
All commits must be signed off to certify you have the right to submit the code:
```bash
git commit -s -m "Your commit message"
```
This adds: `Signed-off-by: Your Name <your@email.com>`

This certifies your agreement with the Developer Certificate of Origin (DCO).

## Code Review Process

After you submit a PR:

1. **Maintainer review**: NCCL engineers will review your code
2. **Discussion**: We may ask questions or request changes
3. **Iteration**: Address feedback and update your PR
4. **Approval**: Once approved, we'll merge your contribution

Please be patient during review. We aim to provide initial feedback within a week. Feel free to ping us if we take much longer than that.

## Communication

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and discussion

## Getting Help

If you need help at any stage:
- Comment on your issue or PR
- Ask questions in GitHub Issues
- Refer to NCCL documentation and examples

We're committed to helping contributors succeed!

## Thank You

We truly appreciate your time and effort in contributing to NCCL.

Happy coding!

