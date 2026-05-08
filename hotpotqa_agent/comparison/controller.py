from .memory import ComparisonMemory
from .schema import ComparisonSchema, ComparisonState
from .tools import ComparisonTools


class ComparisonController:
    def __init__(self, tools: ComparisonTools):
        self.tools = tools

    def step(self, schema: ComparisonSchema, memory: ComparisonMemory) -> ComparisonSchema:
        if schema.state == ComparisonState.SCHEMA_CREATED:
            schema = self.tools.collect_and_extract(schema)
            memory.trace.append({"event": "values_extracted", "schema": schema.to_dict()})
            return schema

        if schema.state == ComparisonState.VALUES_EXTRACTED:
            schema = self.tools.compare(schema)
            memory.trace.append({"event": "compared", "schema": schema.to_dict()})
            return schema

        if schema.state == ComparisonState.COMPARED:
            schema.state = ComparisonState.FINISHED
            return schema

        return schema
