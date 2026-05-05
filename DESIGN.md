# Design Notes

## 1. 为什么不能只做普通 RAG？

普通 RAG 通常是：

```text
Question → Retrieve once → Generate answer
```

但 HotpotQA 的问题往往需要组合多个证据句。例如：

```text
What nationality was James Henry Miller's wife?
```

需要：

```text
James Henry Miller → Ewan MacColl
Ewan MacColl → Peggy Seeger
Peggy Seeger → American
```

因此，一次性检索并不总是能够清楚体现推理过程。更合理的做法是让 Agent 逐步查找证据并更新状态。

## 2. 为什么不强制分类 bridge / comparison？

HotpotQA 提供 `type` 字段，但真实问答场景中用户不会告诉系统题型。并且部分样本可能同时涉及别名识别、概念归一化、数值比较和关系追踪，强制分类会让系统变成模板匹配。

因此，本项目采用统一的状态驱动 Agent：

```text
每一跳判断：
- 当前知道什么？
- 当前缺什么？
- 下一步应该查什么？
- 当前证据是否足够？
```

## 3. Agent 状态设计

核心状态包括：

```text
question
known_facts
evidence_chain
missing_information
agent_trace
answer_ready
final_answer
confidence
```

其中 `evidence_chain` 记录每一个中间事实对应的 HotpotQA title/sentence 证据来源。

## 4. 与已有工作的关系

本项目不是声称提出全新的多跳推理理论，而是进行任务适配：

- ReAct 解决“Agent 如何行动”；
- IRCoT 解决“如何边检索边推理”；
- HGN 启发“证据之间应组织成链或图”；
- HotpotQA 的 supporting_facts 提供 title/sentence 级证据监督。

本项目综合这些思想，构建一个轻量的 HotpotQA evidence-state Agent。
