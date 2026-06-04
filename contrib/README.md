# Contribution Policy for `contrib/`

This directory hosts community and partner contributions that extend NCCL with
new capabilities, algorithms, or higher-level APIs.  The content here is
developed and maintained by its respective contributors, not by the NCCL core
team, and falls outside the NCCL release quality standards.

## Purpose

`contrib/` is the right home for additions that:

- Build on top of NCCL's **public** APIs (`nccl.h`, `nccl_device.h`) without
  modifying NCCL core (no change in `src/`)
- Provide new collective algorithms, higher-level communication APIs, tools,
  or reference implementations

Changes to NCCL core (anything under `src/`) should follow the standard
contribution process described in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Current contributions

| Directory | Description |
|-----------|-------------|
| [`custom_algos/`](custom_algos/) | Reference custom collective kernels built on the NCCL Device API |
| [`nccl_checkpoint/`](nccl_checkpoint/) | NCCL Checkpoint library for multi-node, communicator-aware checkpoint and restore |
| [`nccl_ep/`](nccl_ep/) | NCCL Expert Parallelism (EP) API for MoE communication (dispatch/combine primitives) |
| [`nccl_ubx/`](nccl_ubx/) | UB-X (Ultra Bandwidth — eXperimental): low-latency NVLink collectives with compute fusion (residual + RMSNorm, mxfp8 dispatch) on a symmetric allocator |
| [`nccl_xfer/`](nccl_xfer/) | NCCL Cross-group Transfer (Xfer) API for RL communication (reshard primitives) |

## Upstreaming to contrib/

Before submitting a new contribution, open a GitHub issue to discuss your
proposal.  Include a brief description of what you are adding and why it
belongs in this repository rather than a separate project.  The NCCL team will
provide feedback on whether the contribution is a good fit.

### Requirements for acceptance

A pull request adding a new `contrib/` entry must satisfy all of the following:

1. **Self-contained directory** — all code lives under a single subdirectory
   (`contrib/<name>/`).  No modifications to files outside that directory.

2. **README** — a `README.md` explaining the purpose, build instructions,
   usage, and any hardware or software prerequisites.

3. **Named maintainer(s)** — a `## Maintainers` section in the sub-directory's
   `README.md` listing at least one GitHub username responsible for ongoing
   maintenance.  The maintainer does not have to be a NCCL team member.

4. **License** — the contribution must be compatible with NCCL's
   [Apache 2.0 license](../LICENSE.txt).  Third-party dependencies must be
   documented in `ThirdPartyNotices.txt` or an equivalent file within the
   directory.

5. **Build and test** — the contribution must build cleanly without warnings
   against a current NCCL release.  A basic functional test or benchmark is
   strongly encouraged.

6. **No internal dependency** — code must rely only on NCCL's public headers
   (e.g., `nccl.h`, `nccl_device.h`).  Depending on internal NCCL headers
   or unexported symbols is not supported and could break without notice.

### Review process

Pull requests are reviewed by the NCCL team.  Unlike core changes, the review
focuses primarily on:

- Correctness and absence of obvious bugs
- Adherence to the requirements above
- Risk of confusion with NCCL's official behavior (e.g., naming that could
  imply core NCCL endorsement)

The NCCL team may request changes to scope, naming, or structure before
accepting.

## Maintenance responsibilities

Each contribution has **one or more designated maintainers** responsible for:

- Keeping the code building against current NCCL releases
- Responding to bug reports and pull requests within a reasonable time
- Updating the contribution when NCCL public APIs change in ways that affect it

The NCCL core team does **not** maintain `contrib/` entries and is not
obligated to fix breakage caused by NCCL changes.  When a breaking NCCL change
is anticipated, the team will make a best-effort attempt to notify affected
`contrib/` maintainers in advance.

### Removal policy

A contribution may be removed if:

- It fails to build against a current NCCL release and the maintainer does not
  respond to a removal warning within **four weeks**
- The maintainer explicitly requests removal
- The contribution is found to violate licensing requirements or contains
  security issues that cannot be resolved promptly

Removal will be preceded by a notice on the relevant issue or pull request.
