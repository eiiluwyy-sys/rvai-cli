# RVAI CLI V0.1

## 目标

建立面向 RISC-V AI 模型运行的统一 CLI 原型，
完成模型清单读取、模型详情查看和运行计划生成。

## 支持命令

rvai list
rvai show <model>
rvai run <model> --dry-run

## 本阶段支持的模型

1. qwen-small-int4
2. mobilenet-int8
3. builtin-gemm-int8

## 本阶段不实现

- 真实模型推理
- RISC-V 实机部署
- NPU Runtime
- AI Token
- 性能采集
- 模型下载

## 验收条件

1. rvai list 可以列出三个模型
2. rvai show 可以显示完整 Manifest
3. rvai run --dry-run 可以生成结构化运行计划
4. 不存在的模型能够返回明确错误
5. pytest 测试全部通过
