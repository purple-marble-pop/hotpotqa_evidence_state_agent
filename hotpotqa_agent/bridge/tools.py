import re
from typing import List

from hotpotqa_agent.core.llm import LLMClient
from hotpotqa_agent.core.search import ContextSearchTool, format_evidence_list

from .schema import (
    AttributeStatus,
    BridgeEntitySchema,
    BridgeState,
    CandidateEntity,
)


VERIFY_SYSTEM = """You verify whether a candidate entity satisfies attribute constraints.

Given a candidate, attributes, and evidence, mark each attribute as verified or failed.
Use only the evidence. Do not invent facts.
For origin-location attributes, verify only evidence that the target entity/form originated in that place.
Do not verify using a performer's hometown, band location, recording location, or current residence.

Return ONLY valid JSON:
{
  "verified": true,
  "confirmed_entity": "entity name",
  "attribute_results": [
    {"attribute": "...", "status": "verified | failed", "evidence_ref": "...", "evidence_text": "..."}
  ],
  "reason": "brief reason"
}
"""


CANDIDATE_EXTRACT_SYSTEM = """You extract one candidate for a Bridge Entity Schema.

Inputs:
- schema object: the expected type of candidate, such as person, film, company, location, date, count, nationality, or other answer value.
- schema attributes: evidence-checkable constraints the candidate should satisfy.
- evidence: retrieved sentences from the current HotpotQA context.

Task:
Find the most plausible candidate for the schema. This is candidate discovery, not final verification.
The later verification step will decide whether the candidate satisfies all attributes.

General rules:
- Use only the provided evidence.
- Return exactly one candidate when the evidence gives a plausible candidate; otherwise return found=false.
- The candidate can be a named entity or a literal answer value, depending on the schema object.
- If the schema object says "answer value for relation", extract the requested relation value
  from evidence. Do not return the source/intermediate entity itself unless that entity is
  explicitly the requested value.
- Preserve relation direction. If the schema attribute describes a missing slot, extract the
  entity/value that fills that slot from evidence.
  Examples:
  - Attribute "chemical in which Cadmium Chloride is slightly soluble" and evidence
    "Cadmium chloride is slightly soluble in alcohol" => candidate_entity = "alcohol".
  - Attribute "person whom Milhouse was named after" and evidence
    "Milhouse was named after President Richard Nixon's middle name" =>
    candidate_entity = "President Richard Nixon".
  - Attribute "city where The Oberoi Group has its head office" and evidence
    "The Oberoi Group has its head office in Delhi" => candidate_entity = "Delhi".
- For attributes containing "in which", "whom", "where", "that", or "called", prefer the
  relation filler stated in evidence over the page title or the source entity.
- Do not choose a candidate merely because it appears in a page title or shares words with the query.
- Page titles may identify the entity described by the sentence, including pronouns or descriptions in that page.
  If an evidence item is "Peggy Seeger[1]: She ... was married to ... Ewan MacColl",
  then the candidate person satisfying "wife of Ewan MacColl" is "Peggy Seeger".
  If an evidence item is "Cadmium chloride[1]: It is ... slightly soluble in alcohol",
  then the candidate chemical satisfying "chemical in which Cadmium Chloride is slightly soluble"
  is "alcohol", not the page title.
- Prefer candidates supported by more schema attributes.
- If different evidence sentences support different attributes, you may combine them when they refer to the same candidate.
- It is acceptable if not all attributes are proven at this stage; note uncertainty in the reason.
- Return found=false only when the evidence provides no plausible candidate for the schema object.
- Prefer the most specific candidate text supported by evidence.
- Preserve the answer wording used by evidence for counts and dates when possible.
- Keep candidate_entity concise: the entity/value only, not an explanatory sentence.

Return ONLY valid JSON:
{
  "found": true,
  "candidate_entity": "entity or value",
  "evidence_ref": "title[sent_id]",
  "evidence_text": "minimal supporting evidence",
  "reason": "brief reason grounded in the evidence"
}
"""


HIDDEN_BRIDGE_SYSTEM = """You revise a Bridge Entity Schema using hidden bridge evidence.

The normal search could not find a candidate. Look for hidden bridge clues such as alias,
stage name, birth name, also known as, based on, adapted from, character, spouse, or renamed.

If the evidence reveals that an entity in the attribute should be replaced by a hidden bridge
entity, rewrite the attributes while preserving the original target relation.
Only revise an entity when the evidence explicitly states an identity/alias/renaming/adaptation
relationship for the exact entity in the current attributes. Never replace an entity just because
another title shares a word or phrase.

Example:
- Attribute: "wife of James Henry Miller"
- Evidence: "James Henry Miller, better known by his stage name Ewan MacColl..."
- Revised attribute: "wife of Ewan MacColl"

Return ONLY valid JSON:
{
  "revised": true,
  "revised_attributes": ["..."],
  "hidden_bridge_entity": "entity name",
  "evidence_ref": "title[sent_id]",
  "evidence_text": "supporting sentence",
  "reason": "brief reason"
}
"""


class AttributeBridgeTools:

    def __init__(self, example, llm: LLMClient, top_k: int = 5):
        self.search_tool = ContextSearchTool(example)
        self.llm = llm
        self.top_k = top_k

    def _attribute_texts(self, schema: BridgeEntitySchema, include_relaxed: bool = True) -> List[str]:
        if include_relaxed:
            return [item.description for item in schema.attributes]
        return schema.hard_attribute_texts()

    def _supports_hidden_revision(self, schema: BridgeEntitySchema, evidence_text: str) -> bool:
        text = evidence_text.lower()
        hidden_markers = [
            "better known as",
            "also known as",
            "stage name",
            "birth name",
            "renamed",
            "based on",
            "adapted from",
        ]
        if not any(marker in text for marker in hidden_markers):
            return False

        mentioned_entities = []
        for attr in schema.attributes:
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", attr.description)
            mentioned_entities.extend(a or b for a, b in quoted)
            # Keep this simple and conservative: title-like chunks after relation words.
            for marker in (" of ", " in ", " on ", " by ", " to "):
                if marker in attr.description:
                    mentioned_entities.append(attr.description.split(marker, 1)[1].strip())

        return any(entity and entity.lower() in text for entity in mentioned_entities)

    def _search_schema_evidence(self, schema: BridgeEntitySchema):
        hard_attrs = schema.hard_attribute_texts()
        all_attrs = self._attribute_texts(schema)

        evidence_by_ref = {}
        if schema.object.startswith("answer value for relation:"):
            for attr in all_attrs:
                entity = self._entity_from_relation_attribute(attr)
                if entity:
                    for item in self.search_tool.lookup_title(entity, top_k=self.top_k):
                        evidence_by_ref[item.ref] = item
        for attr in hard_attrs or all_attrs:
            for item in self.search_tool.search(attr, top_k=self.top_k):
                evidence_by_ref[item.ref] = item
        return list(evidence_by_ref.values())

    def _entity_from_relation_attribute(self, attribute: str) -> str:
        for marker in (" of ", " for ", " by ", " in ", " at "):
            if marker in attribute:
                return attribute.rsplit(marker, 1)[1].strip()
        return ""

    def _search_candidate_attribute_evidence(
        self, candidate_name: str, attributes: List[str]
    ):
        evidence = []
        seen_refs = set()

        for attr in attributes:
            query = " ".join([candidate_name, attr]).strip()
            for item in self.search_tool.lookup_title(candidate_name, top_k=self.top_k):
                if item.ref in seen_refs:
                    continue
                seen_refs.add(item.ref)
                evidence.append(item)
            for item in self.search_tool.search(query, top_k=self.top_k):
                if item.ref in seen_refs:
                    continue
                seen_refs.add(item.ref)
                evidence.append(item)

        return evidence

    def _candidate_from_evidence(self, schema: BridgeEntitySchema, evidence) -> CandidateEntity | None:
        if not evidence:
            return None
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        result = self.llm.chat_json(
            CANDIDATE_EXTRACT_SYSTEM,
            (
                f"Schema object: {schema.object}\n"
                f"Schema attributes: {self._attribute_texts(schema)}\n"
                f"Evidence:\n{format_evidence_list(evidence)}"
            ),
        )
        candidate_name = str(result.get("candidate_entity", "")).strip()
        if not result.get("found") or not candidate_name:
            return None
        return CandidateEntity(
            name=candidate_name,
            evidence_ref=str(result.get("evidence_ref", "")).strip(),
            evidence_text=str(result.get("evidence_text", "")).strip(),
            score=float(evidence[0].score if evidence else 0.0),
        )

    def search(self, schema: BridgeEntitySchema) -> BridgeEntitySchema:
        evidence = self._search_schema_evidence(schema)
        schema.candidate_entities = []
        candidate = self._candidate_from_evidence(schema, evidence)
        if candidate:
            schema.candidate_entities.append(candidate)
        schema.state = (
            BridgeState.CANDIDATE_FOUND
            if schema.candidate_entities
            else BridgeState.CANDIDATE_NOT_FOUND
        )
        return schema

    def hidden_search(self, schema: BridgeEntitySchema) -> BridgeEntitySchema:
        hidden_terms = [
            "alias",
            "also known as",
            "stage name",
            "birth name",
            "based on",
            "adapted from",
            "main character",
            "spouse",
        ]
        query = " ".join([schema.object] + self._attribute_texts(schema) + hidden_terms)
        evidence = self.search_tool.search(query, top_k=self.top_k)
        schema.hidden_bridge_notes = [f"{item.ref}: {item.sentence}" for item in evidence]
        schema.candidate_entities = []

        if evidence and self.llm.enabled:
            revision = self.llm.chat_json(
                HIDDEN_BRIDGE_SYSTEM,
                (
                    f"Schema object: {schema.object}\n"
                    f"Current attributes: {self._attribute_texts(schema)}\n"
                    f"Next relation: {schema.next_relation}\n"
                    f"Evidence:\n{format_evidence_list(evidence)}"
                ),
            )
            raw_revised_attributes = revision.get("revised_attributes") or []
            revised_attributes = [
                str(item).strip()
                for item in raw_revised_attributes
                if str(item).strip()
            ]
            revision_evidence = (
                str(revision.get("evidence_text", "")).strip()
                or format_evidence_list(evidence)
            )
            if (
                revision.get("revised")
                and revised_attributes
                and self._supports_hidden_revision(schema, revision_evidence)
            ):
                for attr, revised in zip(schema.attributes, revised_attributes):
                    attr.description = revised
                    attr.status = AttributeStatus.UNVERIFIED
                    attr.constraint_type = "hard"
                    attr.evidence_ref = str(revision.get("evidence_ref", "")).strip()
                    attr.evidence_text = str(revision.get("evidence_text", "")).strip()
                schema.hidden_bridge_notes.append(str(revision.get("reason", "")).strip())
                schema.state = BridgeState.SCHEMA_CREATED
                return schema

        candidate = self._candidate_from_evidence(schema, evidence)
        if candidate:
            schema.candidate_entities.append(candidate)
        schema.state = (
            BridgeState.CANDIDATE_FOUND
            if schema.candidate_entities
            else BridgeState.CANDIDATE_NOT_FOUND
        )
        return schema

    def verify(self, schema: BridgeEntitySchema) -> BridgeEntitySchema:
        if not schema.candidate_entities:
            schema.state = BridgeState.CANDIDATE_NOT_FOUND
            return schema
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")

        candidate = schema.candidate_entities[0]
        attributes = [item.description for item in schema.attributes]
        evidence_text = format_evidence_list(
            self._search_candidate_attribute_evidence(candidate.name, attributes)
        )
        result = self.llm.chat_json(
            VERIFY_SYSTEM,
            f"Candidate: {candidate.name}\nAttributes: {attributes}\nEvidence:\n{evidence_text}",
        )

        results = result.get("attribute_results", [])
        for attr in schema.attributes:
            matched = next(
                (item for item in results if item.get("attribute") == attr.description),
                None,
            )
            if not matched:
                attr.mark_failed()
                continue
            if matched.get("status") == AttributeStatus.VERIFIED.value:
                attr.mark_verified(
                    evidence_ref=str(matched.get("evidence_ref", "")),
                    evidence_text=str(matched.get("evidence_text", "")),
                )
            else:
                attr.mark_failed()

        if schema.all_hard_attributes_verified():
            confirmed_name = candidate.name
            if schema.next_relation:
                confirmed_name = str(result.get("confirmed_entity") or candidate.name)
            schema.confirmed_entity = CandidateEntity(
                name=confirmed_name,
                evidence_ref=candidate.evidence_ref,
                evidence_text=candidate.evidence_text,
                score=candidate.score,
            )
            if not schema.next_relation:
                schema.final_answer = confirmed_name
            schema.state = BridgeState.VERIFIED
        else:
            schema.state = BridgeState.VERIFICATION_FAILED
        return schema
