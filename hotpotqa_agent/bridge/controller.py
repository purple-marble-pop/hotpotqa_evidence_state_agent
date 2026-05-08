from .memory import EvidenceMemory
from .schema import BridgeEntitySchema, BridgeState
from .tools import AttributeBridgeTools


class StrategyController:
    """State machine that dispatches bridge schemas to the right tool."""

    def __init__(self, tools: AttributeBridgeTools):
        self.tools = tools

    def step(self, schema: BridgeEntitySchema, memory: EvidenceMemory) -> BridgeEntitySchema:
        if schema.state == BridgeState.SCHEMA_CREATED:
            return self.tools.search(schema)

        if schema.state == BridgeState.CANDIDATE_NOT_FOUND:
            return self.tools.hidden_search(schema)

        if schema.state == BridgeState.CANDIDATE_FOUND:
            schema = self.tools.verify(schema)
            if schema.state == BridgeState.VERIFIED:
                memory.add_confirmed_entity(schema)
            return schema

        if schema.state == BridgeState.VERIFIED:
            if schema.next_relation:
                memory.add_confirmed_entity(schema)
            else:
                schema.state = BridgeState.FINISHED
            return schema

        if schema.state == BridgeState.VERIFICATION_FAILED:
            schema.mark_failed_attributes_relaxed()
            if schema.candidate_entities and len(schema.attributes) > 1:
                if schema.all_hard_attributes_verified():
                    schema.confirmed_entity = schema.candidate_entities[0]
                    if not schema.next_relation:
                        schema.final_answer = schema.confirmed_entity.name
                    schema.state = BridgeState.VERIFIED
                    memory.add_confirmed_entity(schema)
                    return schema
                return self.tools.search(schema)
            if not schema.hard_attribute_texts():
                return self.tools.hidden_search(schema)
            return self.tools.search(schema)

        return schema
