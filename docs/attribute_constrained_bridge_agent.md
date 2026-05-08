# State-Driven Attribute-Constrained Bridge Agent

This document describes the new bridge reasoning layer under `hotpotqa_agent/bridge`.

## Core Idea

The agent does not directly follow the first retrieved result. Instead, the planner first builds a `BridgeEntitySchema`:

```text
object + attributes + next_relation + state
```

A searched entity is only a candidate. It becomes a confirmed bridge entity only after all hard attributes are verified by evidence.

## Modules

```text
hotpotqa_agent/bridge/
├── schema.py      # BridgeEntitySchema, AttributeConstraint, states
├── memory.py      # EvidenceMemory for schemas, evidence, confirmed entities
├── planner.py     # BridgeSchemaPlanner: question/memory -> schema
├── tools.py       # search, hidden_search, verify
├── controller.py  # StrategyController state machine
└── __init__.py
```

## State Machine

```text
schema_created
    -> search
    -> candidate_found / candidate_not_found

candidate_not_found
    -> hidden_search
    -> candidate_found / candidate_not_found

candidate_found
    -> verify
    -> verified / verification_failed

verified + next_relation
    -> write confirmed entity to EvidenceMemory
    -> Planner creates the next schema

verified + no next_relation
    -> finished

verification_failed
    -> mark failed attributes as relaxed
    -> search again with remaining hard attributes
```

## Verification Failure Policy

Failed attributes are not deleted. They are kept in the schema and marked as `failed` or `relaxed`.

This lets the agent:

- avoid losing question constraints,
- continue searching when one attribute is hard to match,
- use relaxed attributes later for confidence and reflection,
- trigger hidden bridge search or schema revision if the same constraint keeps failing.

## Example Schema

```json
{
  "object": "film",
  "attributes": [
    {
      "description": "released in 2003",
      "status": "unverified",
      "constraint_type": "hard"
    },
    {
      "description": "has scenes filmed at Quality Cafe in Los Angeles",
      "status": "unverified",
      "constraint_type": "hard"
    }
  ],
  "next_relation": "directed by",
  "state": "schema_created"
}
```

