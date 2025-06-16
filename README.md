# 全量RAG技术：更简单、更实用的实现方法 ✨

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-API-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![Medium](https://img.shields.io/badge/Medium-Blog-black?logo=medium)](https://medium.com/@fareedkhandev/testing-every-rag-technique-to-find-the-best-094d166af27f)

本仓库采用清晰、实用的方法讲解**检索增强生成(RAG)**技术，将高级技术分解为简单易懂的实现。不同于依赖`LangChain`或`FAISS`等框架，这里的所有内容都使用常见的Python库`openai`、`numpy`、`matplotlib`等构建。

目标很简单：提供可读、可修改且具有教育意义的代码。通过关注基础原理，本项目帮助揭开RAG的神秘面纱，使其工作原理更易于理解。

## 更新日志：📢

- (2025年5月12日) 新增关于如何使用知识图谱处理大数据的笔记本
- (2025年4月27日) 新增通过简单RAG+重排序器+查询改写寻找最佳RAG技术的笔记本
- (2025年3月20日) 新增关于强化学习RAG的笔记本
- (2025年3月7日) 仓库新增20种RAG技术实现

## 🚀 内容概览

本仓库包含一系列Jupyter Notebook，每个都专注于特定的RAG技术。每个笔记本提供：

- 技术要点的简明解释
- 从零开始的逐步实现
- 带有内联注释的清晰代码示例
- 展示技术效果的评估与比较
- 结果可视化呈现

以下是涵盖的技术概览：

| 笔记本                                      | 描述                                                                                                                                                         |
| :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [1. 简单RAG](01_simple_rag.ipynb)           | 基础RAG实现，最佳入门选择                                                                                                       |
| [2. 语义分块](02_semantic_chunking.ipynb) | 基于语义相似度拆分文本，生成更有意义的文本块                                                                                           |
| [3. 分块大小选择器](03_chunk_size_selector.ipynb) | 探索不同分块大小对检索性能的影响                                                                                    |
| [4. 上下文增强RAG](04_context_enriched_rag.ipynb) | 检索相邻文本块以提供更多上下文                                                                                                     |
| [5. 上下文分块标题](05_contextual_chunk_headers_rag.ipynb) | 在嵌入前为每个文本块添加描述性标题                                                                                                |
| [6. 文档增强RAG](06_doc_augmentation_rag.ipynb) | 从文本块生成问题以增强检索过程                                                                                           |
| [7. 查询转换](07_query_transform.ipynb)   | 通过重写、扩展或分解查询来改进检索，包含**Step-back Prompting**和**子查询分解**技术                                      |
| [8. 重排序器](08_reranker.ipynb)               | 使用LLM对初始检索结果重新排序以提高相关性                                                                                       |
| [9. 相关片段提取](09_rse.ipynb)                         | 识别并重建连续的文本片段，保持上下文完整性                                                   |
| [10. 上下文压缩](10_contextual_compression.ipynb) | 实现上下文压缩以过滤和压缩检索到的文本块，最大化相关信息                                                 |
| [11. 反馈循环RAG](11_feedback_loop_rag.ipynb) | 整合用户反馈使RAG系统持续学习改进                                                                                      |
| [12. 自适应RAG](12_adaptive_rag.ipynb)     | 根据查询类型动态选择最佳检索策略                                                                                          |
| [13. 自监督RAG](13_self_rag.ipynb)             | 实现Self-RAG，动态决定检索时机和方式，评估相关性和支持度                                        |
| [14. 命题分块](14_proposition_chunking.ipynb) | 将文档分解为原子性事实陈述以实现精确检索                                                                                      |
| [15. 多模态RAG](15_multimodel_rag.ipynb)   | 结合文本和图像进行检索，使用LLaVA为图像生成说明文字                                                                  |
| [16. 融合RAG](16_fusion_rag.ipynb)         | 结合向量搜索与基于关键词(BM25)的检索以获得更好结果                                                                                |
| [17. 图RAG](17_graph_rag.ipynb)           | 将知识组织为图结构，实现相关概念遍历                                                                                        |
| [18. 层次化RAG](18_hierarchy_rag.ipynb)        | 构建层次化索引(摘要+详细分块)实现高效检索                                                                                   |
| [19. HyDE RAG](19_HyDE_rag.ipynb)             | 使用假设文档嵌入改进语义匹配                                                                                              |
| [20. 校正RAG](20_crag.ipynb)                     | 动态评估检索质量并使用网络搜索作为后备方案                                                                           |
| [21. 强化学习RAG](21_rag_with_rl.ipynb)                     | 使用强化学习最大化RAG模型的奖励                                                                           |
| [最佳RAG查找器](best_rag_finder.ipynb)     | 使用简单RAG+重排序器+查询改写为给定查询寻找最佳RAG技术                                                                        |
| [22. 知识图谱处理大数据](22_Big_data_with_KG.ipynb) | 使用知识图谱处理大型数据集                                                                                                                     |

## 🗂️ 仓库结构

```
fareedkhan-dev-all-rag-techniques/
├── README.md                          <- You are here!
├── 01_simple_rag.ipynb
├── 02_semantic_chunking.ipynb
├── 03_chunk_size_selector.ipynb
├── 04_context_enriched_rag.ipynb
├── 05_contextual_chunk_headers_rag.ipynb
├── 06_doc_augmentation_rag.ipynb
├── 07_query_transform.ipynb
├── 08_reranker.ipynb
├── 09_rse.ipynb
├── 10_contextual_compression.ipynb
├── 11_feedback_loop_rag.ipynb
├── 12_adaptive_rag.ipynb
├── 13_self_rag.ipynb
├── 14_proposition_chunking.ipynb
├── 15_multimodel_rag.ipynb
├── 16_fusion_rag.ipynb
├── 17_graph_rag.ipynb
├── 18_hierarchy_rag.ipynb
├── 19_HyDE_rag.ipynb
├── 20_crag.ipynb
├── 21_rag_with_rl.ipynb
├── 22_big_data_with_KG.ipynb
├── best_rag_finder.ipynb
├── requirements.txt                   <- Python dependencies
└── data/
    └── val.json                       <- Sample validation data (queries and answers)
    └── AI_Information.pdf             <- A sample PDF document for testing.
    └── attention_is_all_you_need.pdf  <- A sample PDF document for testing (for Multi-Modal RAG).
```

## 🛠️ 快速开始

1. **克隆仓库:**

    ```bash
    git clone https://github.com/FareedKhan-dev/all-rag-techniques.git
    cd all-rag-techniques
    ```

2. **安装依赖:**

    ```bash
    pip install -r requirements.txt
    ```

3. **设置OpenAI API密钥:**

    - 从[Nebius AI](https://studio.nebius.com/)获取API密钥
    - 将API密钥设置为环境变量:

        ```bash
        export OPENAI_API_KEY='您的Nebius AI API密钥'
        ```

        或(Windows系统):

        ```bash
        setx OPENAI_API_KEY "您的Nebius AI API密钥"
        ```

        或在Python脚本/笔记本中:

        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "您的Nebius AI API密钥"
        ```

4. **运行笔记本:**

    使用Jupyter Notebook或JupyterLab打开任意笔记本文件(`.ipynb`)。每个笔记本都是自包含的，可以独立运行。设计上每个文件内的笔记本应按顺序执行。

    **注意:** `data/AI_Information.pdf`文件提供测试用示例文档，可替换为您自己的PDF。`data/val.json`包含用于评估的示例查询和理想答案。
    'attention_is_all_you_need.pdf'用于测试多模态RAG笔记本。

## 💡 核心概念

- **嵌入(Embeddings):** 文本的数值表示，捕捉语义信息。我们使用Nebius AI的嵌入API，在许多笔记本中也使用`BAAI/bge-en-icl`嵌入模型。

- **向量存储(Vector Store):** 存储和搜索嵌入的简单数据库。我们使用NumPy创建自己的`SimpleVectorStore`类进行高效相似度计算。

- **余弦相似度(Cosine Similarity):** 衡量两个向量相似度的指标，值越高表示相似度越大。

- **分块(Chunking):** 将文本分割为更小、更易管理的部分。我们探索多种分块策略。

- **检索(Retrieval):** 为给定查询寻找最相关文本块的过程。

- **生成(Generation):** 使用大语言模型(LLM)基于检索到的上下文和用户查询生成响应。我们通过Nebius AI的API使用`meta-llama/Llama-3.2-3B-Instruct`模型。

- **评估(Evaluation):** 通过比较参考答案或使用LLM评分来评估RAG系统响应质量。

## 🤝 参与贡献

欢迎贡献代码！
