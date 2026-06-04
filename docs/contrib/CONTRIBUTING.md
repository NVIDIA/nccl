# Contributing to NCCL Documentation

Thank you for your interest in contributing to NCCL documentation! We appreciate the time and
effort you're putting into helping improve the library. This document guides you through
contributing material under **`docs/contrib/`** (community provided documentation).

Also see **`CONTRIBUTING.md`** in the repository root; that guide covers contributions to NCCL source
code, build, tests, and the standard user-facing docs flow.

## Before You Start

To help ensure your contribution can be accepted smoothly:

- **Check existing documentation** — including the [user guide](../userguide/README.md) and
  **`docs/contrib/README.md`** — so your work complements rather than duplicates what is already maintained.
- **Coordinate with us** raise a [github Issue](https://github.com/NVIDIA/nccl/issues/new?template=ISSUE.yaml)
  proposing the documentation. We don't want you to do a lot of work if we have something
  already planned or if the topic is about to undergo significant changes! By contributing documentation
  you are saving us time... let us help save yours!

## What We're Looking For

We welcome **`docs/contrib/`** additions that **extend or specialize** NCCL documentation — for example
tutorials, examples, or teaching material that goes beyond what the bundled user
guide is meant to carry.

**Good fits:**

- **Design and internals notes**: How subsystems behave (within what you can verify from source and
  public behavior), diagrams, glossary-style clarifications.
- **Tutorials / walkthroughs**: Step-by-step notes for a concrete setup (environment, communicator
  lifecycle, symmetric memory / device APIs, transports, profiling hooks, etc.). Prefer linking to or
  building on **`docs/examples/`** rather than inventing standalone code that drifts from the repo.
- **Worked examples + narrative**: Commentary that walks through example code already in-tree (or minimal
  diff extensions), explaining *why* something is structured a certain way or how it maps to real jobs.
- **Operational / deployment topics**: Debugging checklists, environment variables explained in context,
  fabric or topology quirks, performance interpretation — grounded in reproducible setups where possible.
- **Translations or localized guides**: Welcome when they reproduce meaning faithfully **and** state
  **original author, translator, NCCL revision, and license/attribution** clearly at the top of the doc.

**Not a good fit:**
- **Performance marketing** that compares NCCL with something else and draws conclusions that can be
  misleading other users. 

## Making Your Contribution Successful

### Scope and focus

- Keep a single PR centered on **one coherent doc change** (one new doc, or one clear edit theme).
- Large rewrites are easier when split into logical follow-up PRs (e.g., structure pass, then
  technical corrections).

### Accuracy and citations

- **Version and API drift**: Mention the NCCL (and, if relevant, CUDA / driver) context you validated
  against. Link to APIs, structs, or source files when referencing behavior.
- **Claims about behavior**: Prefer citations to **specific files and symbols** in this repository over
  speculation. If behavior is heuristic or undocumented, label it explicitly as such.
- **Third-party quotations, figures, or long excerpts**: Respect license and attribution; summarize or
  link when in doubt.

### Removal policy

A contribution may be removed if it becomes outdated or inaccurate. In such cases, we'll make a reasonable
effort to contact the author and give them an opportunity to update the document.
We also reserve the right to remove a contribution for other reasons without prior notice.

## Getting Started

1. **Fork the repository**: Fork [https://github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl) and clone your fork locally.

2. **Create a branch**: Use a descriptive branch name for your work.
   ```bash
   git checkout -b docs/my-topic
   ```

3. **Add your documentation**: Place files in an appropriate subdirectory of **`docs/contrib/`**. Feel
   free to add a new category folder if none of the existing ones fit.

4. **Update `docs/contrib/README.md`**: List your document with a summary and link so others
   can discover it.

5. **Commit and push**:
   ```bash
   git add <files>
   git commit -s --message "<descriptive message, see Pull Request Guidelines>"
   git push origin <branch-name> --set-upstream
   ```

## Documentation Style

### Markdown

- Prefer **semantic headings** (`##`, `###`) that match outline depth you would expect in rendered HTML.
- **Links**: Use descriptive link text and stable URLs (arXiv, GitHub blobs/tags, NVIDIA docs).
- **Tables**: Real Markdown tables, not one-column fake “tables” hiding prose (they read poorly on GitHub
  and in narrow viewports).

### Naming and placement

- **Filenames**: Prefer `Mixed_Case_With_Underscores.md`, but stay consistent within
  a subdirectory.
- **`README.md`** at `docs/contrib` only, not in the subdirectories.

### Language and inclusivity

- NCCL readers are an international audience! If you are writing documentation in English, then write
  for all readers: write simply, avoid unexplained jargon
  and metaphors, and don't use any cultural references that won't be comprehensible to all.

## Pull Request Guidelines

### Before submitting

1. **Rebase** (or merge) onto the latest default branch tip you plan to land on.
2. **Self-review**: Read the Markdown in a renderer (GitHub preview or local tools). Confirm links work
   and formatting is as intended.

### Pull request description

- Summarize **what** readers get from the doc and **who** it is for (beginner, integrator,
  internals-focused, etc.).
- List **limitations** (“validated on …”, “assumes GPUDirect RDMA”) so reviewers match expectations.

### Commit messages

- Use **imperative mood**: “Add …”, “Expand …”, “Fix wording in …”.
- First line: short summary (**~72 characters** is reasonable for Git logs).
- After a blank line, add context: problem (gap in contrib docs), what you added, caveats.

Example:

```text
Add contrib guide for symmetric memory internals

Adds a conceptual overview under docs/contrib/GIN/, links to allocator and
examples. Validated wording against ncclMemAlloc paths in allocator.cc for NCCL 2.x;
note on proxy vs GDAKI may need refresh when transport matrix changes.

Signed-off-by: ...
```

### Signed commits

Commits should include a **Signed-off-by** line (Developer Certificate of Origin), same as for code
contributions:

```bash
git commit -s -m "Your commit message"
```

This applies to textual contributions as well — you are certifying you have the right to submit the
material under the project's licensing terms. This includes the right for diagrams, graphs, etc.

## Review Process

After you open a PR:

1. Maintainers review for accuracy, scope, duplication, and readability.
2. We may suggest structural edits or ask for links to authoritative sources.
3. Address feedback via additional commits or a force-pushed clarification.
4. Once approved, the change is merged.

Initial feedback timing depends on Maintainer load; commenting on stalled PRs is fine.

## Thank You

We truly appreciate your time and effort in improving NCCL's extended documentation community.
