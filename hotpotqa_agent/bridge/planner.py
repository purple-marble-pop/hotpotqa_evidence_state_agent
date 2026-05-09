from typing import Any, Dict

from .memory import EvidenceMemory
from .schema import BridgeEntitySchema


BRIDGE_PLANNER_SYSTEM = """You are a planner for attribute-constrained bridge reasoning.

Build the next Bridge Entity Schema from the question and current evidence memory.

Schema fields:
- object: the current entity type or object to find, such as film, person, company, song, school, location.
- attributes: all required attributes that identify this entity. Include relation clues and constraints from the question.
- candidate_entities: always [] when planning a new schema.
- confirmed_entity: always null when planning a new schema.
- next_relation: the relation to follow after this entity is verified. Use an empty string if this entity is the final answer.
- state: always "schema_created" when planning a new schema.

Rules:

-Do not output a plain search query.
   The Planner must output a Bridge Entity Schema in valid JSON only.

-The schema should describe what must be verified before an entity can be used for the next hop.

-First identify the wh-target.
   - If the question asks "which/who <described person/entity> ...", build the schema directly for that unknown answer-bearing entity, and put all descriptors as attributes.
   - Example:
     "Gunmen from Laredo starred which narrator of Frontier?"
     object = "person"
     attributes = ["starred in Gunmen from Laredo", "narrator of Frontier"]
     next_relation = "full name"

-Preserve the direction of relations from the question. Do not reverse subject and object.
   When the unknown entity is a slot inside a sentence, write the attribute so the slot remains
   the candidate.
   Example:
   "Cadmium Chloride is slightly soluble in this chemical, it is also called what?"
   object = "chemical"
   attributes = ["chemical in which Cadmium Chloride is slightly soluble"]
   next_relation = "common_name"
   Do not write attributes = ["slightly soluble in Cadmium Chloride"], because that means the
   candidate is soluble in Cadmium Chloride.

-For questions with "this/that/which <type>" referring to a missing relation value, make the
   missing value the candidate entity. Use attributes like:
   "chemical in which X is soluble", "person whom X was named after",
   "city where X has its head office", "company that acquired X".

-If the wh-target is an attribute value such as year, nationality, city, state, county, length, album, or network, first build a schema for the entity that owns this attribute, and put the requested attribute in next_relation.

-Use a work title as an attribute constraint when the unknown entity is a person described by a role in that work.
   Do not first build a schema only to verify a clearly named work title unless the question asks for an attribute of that work.

-For questions asking for an attribute of "X's Y", first build a schema for Y, not X.
   Example:
   "What nationality was James Henry Miller's wife?"
   object = "person"
   attributes = ["wife of James Henry Miller"]
   next_relation = "nationality"
-If a confirmed bridge entity already exists, plan the next entity or relation from that confirmed entity.

-If the latest verified schema has next_relation, do not repeat that same schema.
   Use the confirmed entity plus that next_relation to build the next schema.

-If memory contains a hidden bridge node or alias for the original entity, rewrite the next schema using the confirmed hidden bridge entity.
   Example:
   James Henry Miller = Ewan MacColl
   Then use:
   attributes = ["married to Ewan MacColl"]
   not:
   attributes = ["wife of James Henry Miller"]

-If the question asks for an attribute of the entity reached by next_relation, put that final attribute in next_relation.
    Example:
    after confirming the wife entity, use next_relation = "nationality".

-Keep attributes concise and evidence-checkable.

-Do not invent attributes that are not required by the question.

Return ONLY valid JSON in this exact shape:
{
  "object": "...",
  "attributes": [
    {
      "description": "...",
      "status": "unverified",
      "constraint_type": "hard",
      "evidence": null
    }
  ],
  "candidate_entities": [],
  "confirmed_entity": null,
  "next_relation": "...",
  "state": "schema_created"
}
"""


class BridgeSchemaPlanner:
    """LLM planner that creates Bridge Entity Schema objects."""

    def __init__(self, llm):
        self.llm = llm

    def plan(self, memory: EvidenceMemory) -> BridgeEntitySchema:
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        data: Dict[str, Any] = self.llm.chat_json(
            BRIDGE_PLANNER_SYSTEM,
            f"Current evidence memory:\n\n{memory.compact_context()}\nBuild the next schema.",
        )
        return BridgeEntitySchema.from_planner_json(data)
