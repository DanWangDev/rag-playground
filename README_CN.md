# RAG Playground

[English](README.md)

> 动手学习 RAG（检索增强生成）的交互式平台 — 用 Python 从零构建每一个组件。
> 无需框架，无需云 API — 只有你、Python 和本地大模型。

## 工作原理

```
                          RAG 流水线
   ┌─────────┐    ┌─────────┐    ┌───────────┐    ┌──────────┐
   │  加载    │───▶│  切分    │───▶│  向量化   │───▶│  存储    │
   │  文档    │    │  文本    │    │  嵌入     │    │  向量    │
   └─────────┘    └─────────┘    └───────────┘    └─────┬────┘
                                                        │
   ┌─────────┐    ┌──────────┐    ┌───────────┐         │
   │  回答   │◀───│  生成    │◀───│  增强提示  │◀────────┘
   │         │    │  (LLM)   │    │           │    检索 Top-K
   └─────────┘    └──────────┘    └───────────┘    相关片段
                                                        ▲
                                              ┌─────────┴─────────┐
                                              │  用户提问          │
                                              │  "什么是 RAG？"    │
                                              └───────────────────┘
```

## 快速开始

**前置条件：** Python 3.11+ 和 [Ollama](https://ollama.com)

```bash
# 1. 安装 Ollama
#    macOS:   brew install ollama
#    Windows: winget install Ollama.Ollama
#    Linux:   curl -fsSL https://ollama.com/install.sh | sh

# 2. 启动 Ollama（保持运行）
ollama serve

# 3. 克隆项目，安装依赖并拉取模型
git clone https://github.com/DanWangDev/rag-playground.git
cd rag-playground
pip install -e ".[dev]"
python scripts/pull_models.py
python scripts/setup.py

# 4. 运行第一个练习
python -m rag_playground.m02_data_loading.exercise
```

## 学习路径（8 个模块）

每个模块是一个独立的学习单元，包含概念文档、源代码、交互式练习和测试。

```
 模块 02 ──▶ 模块 03 ──▶ 模块 04 ──▶ 模块 05 ──▶ 模块 06 ──▶ 模块 07 ──▶ 模块 08
  加载        切分         嵌入         存储         检索         流水线       评估
                                                              │
 模块 01 ───────────────────────────────────────────────────┘
   LLM 基础（可并行学习）
```

| # | 模块 | 构建内容 | 核心概念 |
|---|------|---------|---------|
| 01 | **LLM 基础** | 对话 + 流式输出 + 提示工程 | Token、Temperature、系统提示词、思维链 |
| 02 | **数据加载** | 文件加载器（txt, md, 目录） | 文档抽象、元数据、编码 |
| 03 | **文本切分** | 3 种切分策略 + 重叠 | 字符切分、递归切分、Token 切分 |
| 04 | **嵌入向量** | 向量生成 + 相似度计算 | 余弦相似度、最近邻、nomic-embed-text |
| 05 | **向量存储** | 内存 k-NN 搜索引擎 | 暴力搜索、阈值过滤、元数据筛选、持久化 |
| 06 | **检索策略** | 高级检索策略 | 查询扩展、多查询融合、LLM 重排序 |
| 07 | **RAG 流水线** | 完整的端到端 RAG | 检索→增强→生成、流式输出、引用标注 |
| 08 | **质量评估** | 评估指标 | Precision@K、Recall@K、MRR、NDCG、忠实度 |

### 每个模块包含：
- **`concept.md`** — 详细的概念讲解，包含 ASCII 图示、表格和常见陷阱
- **源代码** — 文档齐全的 Python 实现，标注了"关键学习要点"
- **`exercise.py`** — 交互式命令行演练（逐步进行）
- **测试** — 使用 Mock Provider 的单元测试（离线运行，无需 Ollama）

## 架构

```
                    ┌──────────────────────────────┐
                    │        .env  (配置)          │
                    │  CHAT_MODEL=qwen2.5:7b       │
                    │  EMBED_MODEL=nomic-embed-text│
                    └─────────────┬────────────────┘
                                  │
                    ┌─────────────▼────────────────┐
                    │      ModelProvider (抽象)     │
                    │  chat()  chat_stream()        │
                    │  embed()  health_check()      │
                    └─────────────┬────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼─────────┐ ┌──────▼──────┐ ┌─────────▼─────────┐
    │  OllamaProvider   │ │  (未来)     │ │  (未来)           │
    │  本地运行，免费    │ │  OpenAI     │ │  vLLM             │
    └───────────────────┘ └─────────────┘ └───────────────────┘

    零依赖 LangChain、LlamaIndex 或任何 RAG 框架。
    每个组件都从零构建 — 你可以理解每一行代码。
```

## 模型配置

默认模型：**Qwen 2.5 7B**（通过 Ollama 运行）。修改 `.env` 中的一行即可切换：

```bash
CHAT_MODEL=qwen2.5:7b    # 默认（7B 参数，约 5GB 内存）
CHAT_MODEL=llama3.2       # 更轻量、更快
CHAT_MODEL=qwen2.5:14b    # 能力更强，约需 10GB 内存
CHAT_MODEL=mistral:7b      # 替代 7B 方案

EMBED_MODEL=nomic-embed-text       # 768 维，约 274MB
EMBED_MODEL=mxbai-embed-large      # 1024 维，约 669MB
```

未来扩展：实现 `ModelProvider` 抽象类并设置 `CHAT_PROVIDER=openai` 即可接入 OpenAI 兼容接口。

## 技术栈

| 层级 | 技术 | 选择原因 |
|------|------|---------|
| 语言 | Python 3.11+ | RAG/ML 生态的原生语言 |
| LLM 运行时 | Ollama | 一条命令安装，本地运行，免费 |
| HTTP | httpx | 现代异步 HTTP 客户端 |
| 数据校验 | Pydantic | 运行时类型检查 |
| 命令行 | Rich | 美观的终端输出 |
| 服务端 | FastAPI | 异步、自动生成 OpenAPI 文档、内置 SSE |
| 前端 | React + Vite | 交互式 RAG 测试界面 |
| 测试 | pytest | Fixture、异步、参数化 |
| 代码检查 | Ruff | 快速的 Python Linter + Formatter |

## 开发

```bash
pytest                          # 运行全部测试（无需 Ollama）
pytest --cov=rag_playground     # 含覆盖率报告
ruff check src/ tests/          # 代码检查
ruff format src/ tests/         # 代码格式化
```

## License

MIT
