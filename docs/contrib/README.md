# Contributions to NCCL Documentation

The NCCL dev team would like to highlight contributions to the documentation of
the NCCL library. Thank you!

If you would like to contribute, please see [the contributions document](CONTRIBUTING.md).

## Why this exists
Official product documentation has to stay focused and broadly applicable, so it
often can't cover the deep technical detail that helps people really understand
how NCCL works. The documents here fill that gap — exploring specific features
and internal design decisions, usually with concrete examples and practical
context. Whether you're learning the fundamentals or digging into the internals,
these contributions can be a valuable resource.

## Caveats

- **Accuracy is not guaranteed**. Content reflects individual contributors' views
and may contain mistakes or become outdated as NCCL evolves.
- **The NCCL team only coordinates and lightly audits**. We organize and generally
review these documents, but do not validate every detail. Inclusion here does
not carry the guarantees of official product documentation.
- **When in doubt, trust the official sources**. If anything here conflicts with
the official docs or source code, the official sources take precedence.

## Categories

### GIN (GPU-Initiated Networking)

* [NCCL GIN and Symmetric Memory](GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md) by Anonymous, translated from Chinese.

### Frameworks

* [DeepEPv2 Analysis (Part 1)](Frameworks/DeepEPv2_Analysis_Part1/DeepEPv2_Analysis_Part1.md) by Anonymous, translated from Chinese.

### Architecture and Internals

* [NCCL Architecture Learning Guides (bilingual: English and 中文)](architecture/learning/index.md) by tianhao909 (community contribution, not from the NCCL team). A source-oriented deep dive into communicator initialization, topology discovery and search, the performance/tuning model, transport selection, and collective execution. Based on NCCL v2.30u1 (2.30.6).
