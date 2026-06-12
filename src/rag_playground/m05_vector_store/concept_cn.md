# 模块 05：构建向量存储

## 你将学到

- 如何从零构建一个**内存向量存储**
- **k-NN 搜索**如何找到最相似的向量
- **元数据筛选**如何缩小搜索结果范围
- **持久化**如何将向量存储保存到磁盘并重新加载

## 为什么向量存储很重要

没有存储的嵌入只是数字。要为查询检索相关文档，你需要：

1. **存储**嵌入及其关联的片段和元数据
2. **搜索**与查询向量最相似的嵌入
3. **返回**与这些嵌入对应的片段

这就是 RAG 中的"R"——检索。向量存储是实现检索的引擎。

## 内存向量存储的工作原理

### 数据结构

```
InMemoryVectorStore
├── vectors: list[list[float]]     ← 嵌入向量
├── chunks: list[Chunk]            ← 关联的片段
├── metadata: list[dict]           ← 每条目的元数据
└── index: dict[str, int]          ← chunk_id → 位置查找
```

所有列表是平行的：`vectors[i]` 对应 `chunks[i]` 和 `metadata[i]`。

### 添加条目

```python
store = InMemoryVectorStore(embedding_model="nomic-embed-text", dimensions=768)
store.add(embedding, chunk, metadata)
```

### 搜索（暴力 k-NN）

给定查询向量，存储会：
1. 计算与每个已存向量的余弦相似度
2. 按相似度降序排列
3. 返回 top-k 结果

复杂度：O(n × d)，其中 n = 向量数量，d = 维度。
对于 768 维向量和 1000 个条目：约 768K 次操作——在内存中完全可行。
对于 100 万条目：约 768M 次操作——你需要近似索引（ANN）。

### 元数据筛选

预筛选：仅搜索匹配特定元数据字段的向量。
后筛选：搜索所有向量，然后按元数据筛选结果。

我们的实现使用后筛选——更简单，对小规模集合足够。

### 持久化

保存为 JSON：
```python
store.save_to_file("my_store.vectorstore.json")
```

从 JSON 加载：
```python
store = InMemoryVectorStore.load_from_file("my_store.vectorstore.json")
```

## 关键概念

### 暴力搜索 vs 近似搜索

| 方法 | 精确度 | 速度 | 适用场景 |
|------|-------|------|---------|
| 暴力 k-NN | 100% | O(n×d) | 小集合（<10K） |
| HNSW（近似） | ~99% | O(log n) | 大集合（>10K） |
| IVF（近似） | ~95% | O(log n) | 超大集合（>1M） |

我们实现暴力搜索。它精确、易于理解，学习阶段足够快。

### 相似度阈值

筛选结果，仅返回高于最低相似度分数的片段：

```python
results = store.search(query_vector, top_k=5, min_score=0.7)
```

当知识库中没有相关信息时，这可以防止检索器返回不相关的结果。

## 常见陷阱

1. **维度不匹配**：向 768 维存储添加 384 维向量会报错。存储在初始化时固定了维度。
2. **内存使用**：每个 768 维浮点向量约 3KB。10K 向量 ≈ 30MB。大规模集合需监控内存。
3. **无并发**：InMemoryVectorStore 默认不是线程安全的。多线程使用需添加锁。
4. **JSON 精度**：将浮点数保存为 JSON 可能损失微小的精度。生产环境应使用二进制格式。
