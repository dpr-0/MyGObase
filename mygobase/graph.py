import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional, Self, Set

import networkx as nx
import sqlite_vec

from mygobase import DB_PATH
from mygobase.llmtools.embedding import EmbeddingExtractor
from mygobase.llmtools.er import Entity, EntityRelationExtractor, Relations
from mygobase.ner import Content, Contents


@contextmanager
def mygobase_connect():
    with sqlite3.connect(DB_PATH) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        yield conn


class MyGOKnowledgeGraph:
    def __init__(
        self, graph: nx.MultiDiGraph, entities: Dict[Entity, Contents]
    ) -> None:
        self.graph = graph
        self.entities = entities
        self.embedder = EmbeddingExtractor()

    @classmethod
    def from_db(cls, path: str = "db/mygo.db") -> Self:
        entities: Dict[Entity, Contents] = {}
        with mygobase_connect() as conn:
            for entity, content in conn.execute(
                "SELECT entity, content FROM entity"
            ).fetchall():
                entities[entity] = Contents.model_validate_json(content)
            graph = nx.MultiDiGraph()
            for (relations_json_str,) in conn.execute("SELECT ner FROM ner").fetchall():
                relations = Relations.model_validate_json(relations_json_str)
                for relation in relations.relations:
                    try:
                        assert relation.source in entities, relation.source
                        assert relation.target in entities, relation.target
                    except AssertionError:
                        continue
                    else:
                        graph.add_node(
                            relation.source, contents=entities[relation.source]
                        )
                        graph.add_node(
                            relation.target, contents=entities[relation.target]
                        )
                        graph.add_edge(
                            relation.source, relation.target, relation=relation.relation
                        )
        return cls(graph, entities)

    def search_entity(self, entity: Entity) -> Optional[Contents]:
        if entity in self.graph:
            return self.graph.nodes[entity]["contents"]
        return None

    def search_similar_entity(
        self, query: str, k: int = 3, threshold: float = 0.8
    ) -> List[Entity]:
        query_embedding = self.embedder.extract(query)
        with mygobase_connect() as conn:
            res = conn.execute(
                """
                SELECT
                    entity,
                    1 - vec_distance_cosine(embedding, ?) as score
                FROM 
                    entity_embedding
                ORDER BY 
                    score DESC
                LIMIT 
                    ?
                """,
                (sqlite_vec.serialize_float32(query_embedding), k),
            ).fetchall()
        res = filter(lambda x: x[1] >= threshold, res)
        return [entity for entity, _ in res]


class Context:
    def __init__(self) -> None:
        self.contents: Set[Content] = set()

    def add(self, c: Content):
        self.contents.add(c)

    def generate_context(self):
        res = ""
        for c in self.contents:
            res += f"#{c.title}\n\n{c.content}\n\n"
        return res

    def remove(self, c: Content):
        self.contents.remove(c)


class Retriever:
    def __init__(self) -> None:
        self.graph = MyGOKnowledgeGraph.from_db()
        self.er_extractor = EntityRelationExtractor()

    def fast_retrieve(self, query: str) -> Context:
        similar_entities = self.graph.search_similar_entity(query, k=5, threshold=0.55)
        context = Context()
        for similar_entity in similar_entities:
            contents = self.graph.search_entity(similar_entity)
            if contents is None:
                continue
            for c in contents.contents:
                context.add(c)
        return context

    def associate_retrieve(self, query: str) -> Context:
        entities = set()
        for e in self.er_extractor.extract_entities(query).entities:
            if e in self.graph.graph:
                entities.add(e)
            else:
                res = self.graph.search_similar_entity(e, k=1, threshold=0.8)
                if res:
                    entities.add(res[0])
        associate_querys = []
        for source, target, relation in self.graph.graph.subgraph(entities).edges(
            data=True
        ):
            associate_querys.append(f"{source} -> {relation['relation']} -> {target}")

        context = Context()
        for each_query in associate_querys + [query]:
            for entity in self.graph.search_similar_entity(
                each_query, k=5, threshold=0.55
            ):
                contents = self.graph.search_entity(entity)
                if contents is None:
                    continue
                for c in contents.contents:
                    context.add(c)
        # print(context.generate_context())
        return context


if __name__ == "__main__":
    query = "爽世为何同意愛音加入乐团？"
    retriever = Retriever()
    retriever.associate_retrieve(query)
