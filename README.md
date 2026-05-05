# HotpotQA Evidence-State Multi-Hop Agent

本项目是一个面向 HotpotQA 的 **证据状态驱动多跳推理 Agent**。

参考：

- ReAct 的 `Thought → Action → Observation → Finish` 智能体循环；
- IRCoT 的“检索与推理交错进行”思想；
- HotpotQA 自身的 `context` 与 `supporting_facts` 证据结构；

构建一个能够在多跳问答中维护 `known_facts`、`evidence_chain`、`missing_information` 和 `agent_trace` 的轻量级 Agent 框架。

## 系统框架

<img src="assets/system_architecture.png" alt="" width="100%">

## 项目目标

给定一条 HotpotQA 样本：

```text
question + context pages
```

Agent 需要：

1. 根据问题判断当前缺失的信息；
2. 在候选 context 页面中执行 Search / Lookup；
3. 从观察结果中抽取中间事实；
4. 更新结构化 reasoning state；
5. 判断证据是否充分；
6. 若充分则 Finish，否则继续下一跳；
7. 输出最终答案、证据链和每一跳轨迹。

## 项目结构

```text
hotpotqa_evidence_state_agent/
├── hotpotqa_agent/
│   ├── data/
│   │   ├── load_hotpotqa.py
│   │   ├── explore_dataset.py
│   │   └── export_evidence.py
│   └── agent/
│       ├── state.py
│       ├── tools.py
│       ├── llm.py
│       ├── planner.py
│       ├── interpreter.py
│       └── hotpot_agent.py
├── examples/
│   └── run_sample_agent.py
├── requirements.txt
├── .env.example
└── DESIGN.md
```

## 环境准备

推荐使用 conda：

```bat
conda create -n hotpot-agent python=3.11 -y
conda activate hotpot-agent
pip install -r requirements.txt
```

项目需要在根目录创建 `.env` 并配置大模型。

```env
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek-ai/DeepSeek-V3
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024
```

## 查看 HotpotQA 样本

```bash
python -m hotpotqa_agent.data.explore_dataset --split validation --sample-index 0 --sample-count 1
```

如果你已经缓存了数据集，可以加：

```bash
--hf-cache-dir <drive:>/hf_cache/hotpotqa --offline
```

## 导出问题和标准证据

```bash
python -m hotpotqa_agent.data.export_evidence --split train --sample-index 0 --sample-count 100 --output outputs/hotpot_train_evidence_100.md
```

## 运行 Agent 示例

### 1-hop Example

```bash
python examples/run_sample_agent.py --split train --sample-index 7 --max-hops 4
```

```text
Question:
Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.

Gold answer:
Badr Hari

Hop 1:
Action: search {"query": "Badr Hari kick boxer controversies unsportsmanlike conduct crimes"}
Observation:
Badr Hari[2] directly states that he was once considered the best kickboxer in the world and was involved in controversies related to unsportsmanlike conduct and violent crimes outside the ring.
Extracted fact:
Badr Hari was once considered the best kickboxer in the world and has been involved in controversies relating to unsportsmanlike conduct and crimes of violence.

Final answer:
Badr Hari
```

### 2-hop Example

```bash
python examples/run_sample_agent.py --split train --sample-index 8 --max-hops 4
```

```text
Question:
The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?

Gold answer:
2006

Hop 1:
Action: search {"query": "House of Anubis based on"}
Observation:
House of Anubis[0] states that House of Anubis is based on the Dutch-Belgian television series "Het Huis Anubis".
Extracted fact:
House of Anubis is based on the Dutch-Belgian series "Het Huis Anubis".

Hop 2:
Action: lookup_title {"title_query": "Het Huis Anubis"}
Observation:
Het Huis Anubis[1] states that the series first aired in September 2006.
Extracted fact:
Het Huis Anubis first aired in September 2006.

Final answer:
2006
```

Another 2-hop example:

```bash
python examples/run_sample_agent.py --split train --sample-index 15 --max-hops 4
```

```text
Question:
The 1988 American comedy film, The Great Outdoors, starred a four-time Academy Award nominee, who received a star on the Hollywood Walk of Fame in what year?

Gold answer:
2006

Hop 1:
Action: search {"query": "The Great Outdoors 1988 cast four-time Academy Award nominee"}
Observation:
The Great Outdoors (film)[1] lists the cast, and Annette Bening[3] states that she is a four-time Academy Award nominee.
Extracted fact:
Annette Bening starred in The Great Outdoors and is a four-time Academy Award nominee.

Hop 2:
Action: search {"query": "Annette Bening Hollywood Walk of Fame star year"}
Observation:
Annette Bening[4] states that in 2006, she received a star on the Hollywood Walk of Fame.
Extracted fact:
Annette Bening received a star on the Hollywood Walk of Fame in 2006.

Final answer:
2006
```

### 3-hop Example

```bash
python examples/run_sample_agent.py --split train --sample-index 3 --max-hops 4
```

```text
Question:
What nationality was James Henry Miller's wife?

Gold answer:
American

Hop 1:
Action: search {"query": "James Henry Miller wife"}
Observation:
Ewan MacColl[0] states that James Henry Miller was better known by his stage name Ewan MacColl.
Extracted fact:
James Henry Miller is better known by his stage name Ewan MacColl.

Hop 2:
Action: search {"query": "Ewan MacColl wife"}
Observation:
Peggy Seeger[1] states that Peggy Seeger was married to the singer and songwriter Ewan MacColl until his death in 1989.
Extracted fact:
Ewan MacColl (James Henry Miller) was married to Peggy Seeger.

Hop 3:
Action: lookup_title {"title_query": "Peggy Seeger"}
Observation:
Peggy Seeger[0] states that Margaret "Peggy" Seeger is an American folksinger.
Extracted fact:
Peggy Seeger is an American folksinger.

Final answer:
American
```

Another 3-hop example:

```bash
python examples/run_sample_agent.py --split train --sample-index 16 --max-hops 4
```

```text
Question:
What are the names of the current members of American heavy metal band who wrote the music for Hurt Locker The Musical?

Gold answer:
Hetfield and Ulrich, longtime lead guitarist Kirk Hammett, and bassist Robert Trujillo.

Agent trace:
1. search "Hurt Locker The Musical music written by"
   → Finds that the music for Hurt Locker The Musical was written by Metallica and Stephen R. Schwartz.

2. search "Metallica American heavy metal band current members"
   → Confirms that Metallica is an American heavy metal band.

3. lookup_title "Metallica"
   → Finds the current members on the Metallica page: James Hetfield, Lars Ulrich, Kirk Hammett, and Robert Trujillo.

Final answer:
James Hetfield, Lars Ulrich, Kirk Hammett, and Robert Trujillo
```
