<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL 深度学习路径

如果说官方 NCCL 文档告诉你“API 怎么用”，这一组文档就是告诉你
“NCCL 到底是怎么思考和做决定的”。

请先记住一条总主线：NCCL 会先观察机器拓扑，构建通信图，估算每种候
选方案的代价，然后针对当前消息规模与硬件结构，挑出它认为最便宜的那
个方案再执行。

```mermaid
flowchart LR
    A[用户代码 / 框架] --> B[init.cc / collectives.cc]
    B --> C[bootstrap.cc]
    C --> D[topo.cc + paths.cc + search.cc]
    D --> E[tuning.cc + tuner 插件]
    E --> F[enqueue.cc 调度器]
    F --> G[proxy.cc + transport.cc]
    G --> H[src/device/* GPU 内核]
```

## 为什么 NCCL 初看会让人头大

- 对外 API 很少，但内部其实是一套分布式系统运行时。
- “算法” 与 “协议” 是两层不同选择，NCCL 两层都要选。
- 一次 collective 会同时碰到 CPU 线程、GPU kernel、显存注册、网络建
  连、拓扑启发式和性能模型。

这套文档的目标，就是把这座大山拆成一条能走通的楼梯。

## 推荐阅读顺序

| 步骤 | 文档 | 你会收获什么 |
| --- | --- | --- |
| 1 | [quick-start.md](quick-start.md) | 如何构建、运行示例、第一次该看哪些源码 |
| 2 | [architecture.md](architecture.md) | 从宏观上看 communicator、bootstrap、拓扑、planner、transport、kernel |
| 3 | [collective-execution.md](collective-execution.md) | 一次 `ncclAllReduce` 如何走到设备侧执行 |
| 4 | [topology-and-tuning.md](topology-and-tuning.md) | NCCL 如何认识机器并把它变成 ring/tree/NVLS/CollNet |
| 5 | [math-and-performance.md](math-and-performance.md) | 复杂公式怎么变成直觉和小算例 |
| 6 | [source-code-map.md](source-code-map.md) | 真正改代码时，第一步应该开哪些文件 |

## 一段话总结 NCCL

NCCL 绝不只是“一个 ring kernel”。在 communicator 初始化阶段，
`src/init.cc` 负责收集每个 rank 的信息，`src/bootstrap.cc` 负责做带外
交换，`src/graph/` 下的拓扑子系统负责识别机器并搜索通信图，
`src/graph/tuning.cc` 再把这些图转成延迟和带宽估计，随后
`src/enqueue.cc` 在每次 collective 到来时根据这些估计选择算法与协议。
真正的数据搬运和规约，最后才落到 `src/device/` 下的设备侧实现。

## 最值得先记住的源码锚点

| 文件 | 价值 |
| --- | --- |
| `src/init.cc` | communicator 初始化主轴 |
| `src/bootstrap.cc` | 初始化期间的 rank 间元数据交换 |
| `src/collectives.cc` | 对外 collective API 薄封装 |
| `src/enqueue.cc` | 算法/协议选择、chunking、launch 规划中心 |
| `src/graph/topo.cc` | 硬件拓扑图构建 |
| `src/graph/paths.cc` | 路径分类与可达性 |
| `src/graph/search.cc` | ring/tree/NVLS/CollNet 图搜索 |
| `src/graph/connect.cc` | 把图落成 communicator channel |
| `src/graph/tuning.cc` | 启发式性能模型 |
| `src/transport.cc` 与 `src/transport/*` | transport 注册与具体实现 |
| `src/plugin/*` | net/tuner/profiler/env 插件加载 |
| `src/device/*` | GPU 端 primitive 与 collective kernel |

## 你该从哪一页开始

- 第一次读 NCCL 内部：先看 [architecture.md](architecture.md)
- 调算法选型问题：先看 [topology-and-tuning.md](topology-and-tuning.md)
- 盯某个 collective 的性能：先看 [collective-execution.md](collective-execution.md)
- 准备动手改代码：把 [source-code-map.md](source-code-map.md) 一直开着

## 范围说明

这些文档完全以当前分支源码为依据，目标是帮助你建立“源码如何串起来”
的认识。NCCL 内部实现会持续演进，所以这里的函数名和路径更适合作为读
码锚点，而不是永远不变的内部 ABI 承诺。
