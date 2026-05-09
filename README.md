# HotpotQA Logic-Chain Hybrid Agent

本项目面向 HotpotQA 多跳问答任务，包含一个原始 baseline agent，以及一个改进后的逻辑链导向 hybrid agent。

项目的核心问题是：HotpotQA 的答案往往不能从单个句子直接得到，而需要沿着中间实体、属性约束和关系链逐步推理。原始方法更像是在检索结果中不断更新证据状态；改进方法则尝试显式构建推理逻辑链中的连接点，并围绕这些连接点进行检索、验证和推进。

## 总体思想

### Baseline: 证据状态驱动

原始 `hotpotqa_agent/agent` 是一个 Evidence-State Agent。它参考 ReAct 和 IRCoT 的思想，在每一跳中交错执行：

```text
Thought -> Action -> Observation -> State Update -> Finish
```

它维护：

- `known_facts`
- `evidence_chain`
- `missing_information`
- `agent_trace`

这种方式的优点是直接、灵活，通常能较快得到答案。它更偏向 answer-oriented，即围绕当前缺失信息不断检索和归纳，直到模型认为可以回答。

### Proposed: 逻辑链连接点驱动

改进后的方法不直接把第一个检索结果当作下一跳，而是先分析问题中的逻辑结构，显式规划推理链中的连接点。

可以把多跳问题理解成一条逻辑链：

```text
起点实体/条件 -> 中间连接点 -> 下一跳关系 -> 最终答案
```

其中，中间连接点不是随便检索到的实体，而是必须满足问题中的属性约束。例如：

```text
Question:
The 1988 American comedy film, The Great Outdoors, starred a four-time Academy Award nominee,
who received a star on the Hollywood Walk of Fame in what year?

Bridge entity:
object = person
attributes =
  - starred in The Great Outdoors
  - four-time Academy Award nominee
next_relation = year received Hollywood Walk of Fame star
```

只有当候选实体满足这些属性后，它才会成为 confirmed bridge entity，并被用于下一跳推理。

## 方法结构

项目现在包含三套主要组件。

```text
hotpotqa_agent/
  agent/          # 原始 Evidence-State baseline agent
  bridge/         # 属性约束驱动的 bridge reasoning agent
  comparison/     # 针对 comparison 问题的 agent
  core/           # 共享的 LLM、检索和状态工具
  data/           # HotpotQA 数据加载和样本查看工具
  router.py       # 问题类型判别器
```

### Bridge Agent

Bridge Agent 用于处理 HotpotQA 中的 bridge 类型问题。它的核心表示是 `BridgeEntitySchema`：

```json
{
  "object": "person",
  "attributes": [
    {
      "description": "wife of Ewan MacColl",
      "status": "unverified",
      "constraint_type": "hard"
    }
  ],
  "candidate_entities": [],
  "confirmed_entity": null,
  "next_relation": "nationality",
  "state": "schema_created"
}
```

状态流如下：

```text
schema_created
  -> search
  -> candidate_found / candidate_not_found

candidate_found
  -> verify
  -> verified / verification_failed

verified + next_relation
  -> write to EvidenceMemory
  -> build next schema

verified + no next_relation
  -> finished
```

如果普通搜索找不到候选实体，会调用 `hidden_search` 寻找隐藏桥接信息，例如别名、艺名、改编来源等：

```text
James Henry Miller -> Ewan MacColl
House of Anubis -> Het Huis Anubis
Catching Fire -> The Hunger Games
```

如果 `hidden_search` 仍然找不到候选实体，当前样本会直接结束，避免重复调用到最大轮数。

### Comparison Agent

Comparison 类型问题通常不需要复杂桥接链，而是比较两个实体在某个属性上的差异，例如：

```text
Who was inducted into the Rock and Roll Hall of Fame, David Lee Roth or Cia Berg?
```

因此项目单独实现了 `hotpotqa_agent/comparison`，用于：

- 抽取被比较实体
- 抽取比较属性
- 分别查找两个实体的属性值
- 根据规则输出答案

### Hybrid Agent

`run_hybrid_agent.py` 和 `evaluate.py` 会先通过 router 判断问题类型：

```text
bridge -> Bridge Agent
comparison -> Comparison Agent
```

如果数据集中已有 `type` 字段，默认优先使用数据集标注；也可以使用 LLM router。

## 环境配置

建议使用 conda：

```bash
conda create -n hotpot-agent python=3.11 -y
conda activate hotpot-agent
pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```env
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-ai/DeepSeek-V3.2
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024
```

当前实验中推荐使用：

```env
LLM_MODEL=deepseek-ai/DeepSeek-V3.2
```

原因是该模型在本项目的多轮 JSON 输出、schema planning、candidate extraction 和 verification 中较稳定，速度和准确率也比较均衡。

## 查看 HotpotQA 样本

```bash
python -m hotpotqa_agent.data.explore_dataset --split validation --sample-index 0 --sample-count 1
```

导出问题和 supporting facts：

```bash
python -m hotpotqa_agent.data.export_evidence --split train --sample-index 0 --sample-count 100 --output outputs/hotpot_train_evidence_100.md
```

## 运行示例

运行原始 baseline agent：

```bash
python examples/run_sample_agent.py --split train --sample-index 3 --max-hops 4
```

运行 bridge agent：

```bash
python examples/run_bridge_agent.py --split train --sample-index 3 --max-rounds 8
```

运行 comparison agent：

```bash
python examples/run_comparison_agent.py --split train --sample-index 36 --max-rounds 4
```

运行 hybrid agent：

```bash
python examples/run_hybrid_agent.py --split train --sample-index 20 --max-rounds 8
```

运行后，trace 会写入：

```text
outputs/bridge_trace.md
outputs/comparison_trace.md
```

## 评测

评测改进后的 hybrid agent：

```bash
python evaluate.py --split train --sample-index 150 --sample-count 30 --max-rounds 8
```

评测原始 baseline agent：

```bash
python -m hotpotqa_agent.evaluation.evaluate_agent --split train --sample-index 150 --sample-count 30 --max-hops 4
```

逐样本结果会保存到：

```text
outputs/eval_hybrid.jsonl
outputs/eval_agent.jsonl
```

## 评价指标

本项目将指标分为四类。

### Answer Metrics

- `answer_em`: 预测答案与标准答案完全匹配的比例。
- `answer_f1`: 预测答案与标准答案的 token-level F1。
- `BLEU`: 生成答案与标准答案之间的 n-gram 相似度。
- `ROUGE-L`: 基于最长公共子序列的相似度。
- `METEOR`: 综合词形和词序的生成相似度。

### Evidence Metrics

- `sp_f1` / `support_f1`: supporting facts 的 F1，用于衡量模型是否找到了正确证据。

### Joint Metrics

- `joint_f1`: 同时考虑答案和 supporting facts 的综合 F1。

### Process Metrics

- `rounds`: 平均推理轮数。
- `candidate_found_rate`: 候选中间实体搜索成功率。
- `verification_success_rate`: 候选实体验证成功率。
- `confirmed_chain_length`: 平均确认实体链长度。

## 实验现象

原始 baseline agent 通常在 answer-only 指标上更强，因为它更直接面向最终答案生成。

改进后的 hybrid agent 在部分实验中会牺牲少量 answer EM/F1，但能够显著提升 supporting fact F1 和 joint F1。这说明属性约束和显式验证机制可以增强证据 grounding 和多跳推理链的可解释性。

因此，本项目的核心结论不是简单追求最高答案匹配率，而是探索：

```text
如何用显式逻辑链连接点，让多跳问答过程更可验证、更可解释。
```

## Git 注意事项

不要提交 `.env` 和大规模输出文件。建议 `.gitignore` 至少包含：

```text
.env
__pycache__/
*.pyc
outputs/*.jsonl
outputs/*.md
```
