# Singapore Housing Assistant 项目优化方案

## 项目评估总览

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | 8/10 | LangGraph 设计良好，SQLite 持久化已完成 |
| 代码质量 | 8/10 | 结构清晰，关键 bug 已修复，日志完善 |
| 错误处理 | 8/10 | 统一 [ERROR]/[NO_RESULTS] 前缀 + logger |
| 性能 | 7/10 | lru_cache 覆盖 Maps API + Parent Chunk |
| 可扩展性 | 7/10 | 易加工具/语言，核心难改 |
| 测试覆盖 | 6/10 | 73 个单元测试，待补集成/Mock 测试 |
| **综合** | **7.7/10** | **核心功能完善，待补集成测试和 LangSmith** |

---

## 一、关键问题（需立即修复）

### ~~1.1 消息清理 Bug~~ ✅ 已修复
**修复方案：** 只在有 conversation_summary 时才删除消息，否则保留消息作为上下文

### ~~1.2 破坏性索引~~ ✅ 已修复
**修复方案：** 添加命令行参数 `--rebuild` 和 `--append`，默认模式下如果索引存在则报错退出

### ~~1.3 元数据合并 Bug~~ ✅ 已修复
**修复方案：** 合并 chunk 时保留第一个 chunk 的 metadata，不再用 `" -> "` 拼接


### ~~1.4 把输出改成流式输出~~ ✅ 已完成
**方案：** `astream_events` + async generator `bot_respond`，tag 过滤 `aggregate_llm` 节点实现 token 级流式输出

---

## 二、高优先级优化

### ~~2.1 错误处理标准化~~ ✅ 已完成
**方案：** `config.py` 定义 `TOOL_ERROR_PREFIX="[ERROR]"` / `TOOL_NO_RESULTS_PREFIX="[NO_RESULTS]"`，`tools.py`（5处）和 `maps_tools.py`（7处）统一使用英文前缀格式。所有 `except` 块添加 `logger.error(exc_info=True)`，所有无结果分支添加 `logger.warning()`。

### ~~2.2 持久化存储~~ ✅ 已完成
**方案：** `InMemorySaver` → `SqliteSaver`（`checkpoints.db`）。`thread_id` 持久化到 `thread_id.txt`，重启后自动恢复对话，清空对话时生成新 ID。

### ~~2.3 添加日志系统~~ ✅ 已完成
**方案：** config.py 添加 `setup_logging()`，src/ 下所有 print 替换为 logging

---

## 三、性能优化

### ~~3.1 Google Maps API 缓存~~ ✅ 已完成
**方案：** `MapsToolFactory` 内创建 4 个 `@lru_cache(maxsize=100)` 缓存方法：`distance_matrix`、`directions`、`geocode`、`places_nearby`。工具闭包通过缓存方法调用 API，避免重复计费。

### ~~3.2 Parent Chunk 缓存~~ ✅ 已完成
**方案：** 模块级 `@lru_cache(maxsize=128)` 函数 `_read_json_file()` 缓存 JSON 读取，`clear_store()` 时自动 `cache_clear()`。

### ~~3.3 批量 Parent 检索优化~~ ✅ 已完成（之前已有）
**方案：** `load_content_many()` 已实现批量加载 + 去重 + 排序

---

## 四、代码结构优化

### ~~4.1 清理空模块~~ ✅ 已完成
**方案：** 删除 `src/core/` 整个目录和 `src/db/vector_db_manager.py`

### ~~4.2 配置集中化~~ ✅ 已完成
**方案：** 接通 config.py 已有常量 + 新增 MAPS/SUMMARY 常量，tools.py/nodes.py/maps_tools.py 全部引用 config

### ~~4.3 工具工厂抽象~~ ✅ 已完成
**方案：** 创建 `src/rag_agent/base.py`，定义 `BaseToolFactory(ABC)` 抽象基类 + `timed_tool` 耗时装饰器。`ToolFactory` 和 `MapsToolFactory` 均继承基类。

---

## 五、测试体系建设

### ~~5.1 单元测试~~ ✅ 已完成
**方案：** 6 个测试文件，73 个测试用例，覆盖：
```
tests/
├── test_config.py            # 配置常量和参数验证
├── test_graph_state.py       # State reducer 和 Pydantic 模型
├── test_i18n.py              # 多语言翻译和 fallback
├── test_maps_normalize.py    # 地点名称标准化（含 bug 修复）
├── test_parent_store.py      # 存储层 CRUD + 缓存行为
└── test_prompts.py           # Prompt 模板内容验证
```

### 5.2 集成测试
待做：需要 mock 外部 API（Gemini、Google Maps、Qdrant）

### 5.3 Mock 测试
待做：需要 mock 外部 API

---

## 六、可观测性增强

### 6.1 LangSmith 集成
已有配置，确保启用：
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=singapore-housing-assistant
```

### ~~6.2 性能指标~~ ✅ 已完成
**方案：** `base.py` 中定义 `timed_tool` 装饰器，RAG 工具调用自动记录耗时到 `logger.info`。

---

## 七、实施优先级

| 阶段 | 任务 | 状态 |
|------|------|------|
| **P0** | 修复 3 个关键 bug (1.1-1.3) | ✅ 完成 |
| **P1** | 流式输出 (1.4) | ✅ 完成 |
| **P1** | 错误处理标准化 + 日志系统 (2.1, 2.3) | ✅ 完成 |
| **P2** | 持久化存储 + API 缓存 (2.2, 3.1, 3.2) | ✅ 完成 |
| **P3** | 工具工厂抽象 (4.3) | ✅ 完成 |
| **P3** | 单元测试 (5.1) | ✅ 完成 (73 tests) |
| **P3** | 性能指标 (6.2) | ✅ 完成 |
| **P3** | 集成/Mock 测试 (5.2, 5.3) | 待做 |
| **P3** | LangSmith 集成 (6.1) | 待做 |

---

## 八、验证方法

修复后测试：
1. 多轮对话测试（验证消息清理 bug）
2. 重启后继续对话（验证持久化）
3. 重复位置查询（验证缓存命中）
4. 运行 `pytest tests/` （验证测试覆盖）


## 九、Portfolio 增强（面向 AI/LLM 实习）

### 当前项目亮点（面试加分项）

| 亮点 | 面试官关注点 |
|------|-------------|
| LangGraph 多步 workflow | 不是简单 chain，有 summarize → analyze → agent subgraph → aggregate |
| Parent-Child 两级检索 | 理解 context window 问题，不暴力塞文档 |
| Hybrid search（dense + sparse） | 知道纯语义搜索不够，加了 BM25 |
| Query rewriting + 拆分 | 复杂问题拆成多个子问题并行检索，高级 RAG 模式 |
| Human-in-the-loop | 问题不清楚时主动追问，不盲目回答 |
| 流式输出 | token-level streaming，用户体验好 |
| Google Maps 工具集成 | 不只是文本 RAG，结合外部 API |
| 73 个单元测试 | 工程意识 |

### 短板（需要补强）

| 短板 | 面试影响 | 优先级 |
|------|---------|--------|
| 知识库只有 3 个文档 | 面试官一看就知道数据量太小 | 🔴 高 |
| 没有 RAG 评估指标 | 无法回答"准确率多少？" | 🔴 高 |
| README 过时 | 面试官第一印象差 | 🟡 中 |
| 没有 Reranking | 可能被问"为什么不加 Cross-Encoder？" | 🟡 中 |

### ~~9.1 知识库扩充~~ ✅ 已完成

**方案：** 新增 8 个高质量文档（3→11 个），覆盖区域指南、省钱攻略、防骗、交通、水电网络、签证规定。
参考来源：PropertyGuru、99.co、Rentify、ICA、CEA、SP Services 等 2025-2026 年数据。
待运行 `python indexing.py --rebuild` 重建索引。

### 9.2 README 更新 🟡

**问题：** 项目结构图过时（还有已删除的 `src/core/`），缺架构图

**方案：**
- 更新项目结构，反映当前文件布局
- 添加 Mermaid 架构图展示 LangGraph 流程
- 添加 "Engineering Highlights" 部分
- 添加 `pytest tests/` 测试命令
- 修正 git clone URL

### 9.3 RAG 评估系统 🔴

**问题：** 面试必问"你的 RAG 效果怎么样？怎么评估的？"

**大方向：** 创建评估脚本 + 测试数据集，评估指标方向：
- Answer Relevance（回答是否相关）
- Source Attribution（来源引用是否正确）
- Retrieval Precision（检索精度）
- Response Time（响应时间）

具体方案待知识库扩充后再确定。

### 9.4 Reranking（加分项）🟡

**大方向：** 检索结果加 Cross-Encoder 重排序，提高检索质量。

候选方案：
- `cross-encoder/ms-marco-MiniLM-L-6-v2`（轻量）
- Cohere Rerank API（云服务）
- 自定义 rerank 逻辑

具体方案待做时再确定。

### Portfolio 实施优先级

| 顺序 | 任务 | 面试影响 |
|------|------|---------|
| 1️⃣ | ~~9.1 知识库扩充~~ | ✅ 完成（3→11 个文档） |
| 2️⃣ | 9.3 RAG 评估系统 | 🔴 面试必问 |
| 3️⃣ | 9.4 Reranking | 🟡 技术加分项 |
| 4️⃣ | 9.2 README 更新 | 🟡 最后统一更新 |

---

## 备注/技术问题（待研究）

- 文档处理器是否换成 Unstructured？
- 分块策略：当前 RecursiveCharacterTextSplitter，是否换 SemanticChunker？
- Embedding 模型优化空间？
- 向量数据库：当前 Qdrant，Milvus 是否更好？
- LlamaIndex 是否可以替代当前的 RAG 检索层？
- 后续功能：Query Construction、text2SQL 做具体房源搜索
- MultiQueryRetriever 是否可以优化 rewriting？