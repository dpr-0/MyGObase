import sqlite3
from typing import Iterable, List, Self, Set

import networkx as nx

from mygobase.entity_embedd import load_ner
from mygobase.ner import NER
from mygobase.summarizer import CommunityReporter


class MyGOKnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_entity_and_relation(self, ner: NER):
        entites = ner.entities
        for e in entites:
            self.graph.add_node(e.name, type=e.type)
        relations = ner.relations
        for r in relations:
            self.graph.add_edge(
                r.subject,
                r.object,
                relation=r.relation,
            )

    def query(self, entities: Iterable[str]):
        res = []
        sub_graph = self.graph.subgraph(entities)
        for src, dst, data in sub_graph.edges(data=True):
            res.append(f"{src} -> {data['relation']} -> {dst}")
        return "\n".join(res)

    @classmethod
    def from_db(cls, path: str = "db/mygo.db") -> Self:
        ners = load_ner(path)
        mygo_graph = cls()
        for ner in ners:
            mygo_graph.add_entity_and_relation(ner)
        return mygo_graph


if __name__ == "__main__":
    mygo_graph = MyGOKnowledgeGraph.from_db()

    communities: List[Set[str]] = nx.algorithms.community.louvain_communities(
        mygo_graph.graph, seed=123
    )  # type: ignore
    print(len(communities))
    cp = CommunityReporter()
    for c_id, community in enumerate(communities):
        with sqlite3.connect("db/mygo.db") as conn:
            cur = conn.cursor()
            for entity in community:
                cur.execute(
                    """
                    UPDATE entity_embedding_rowids
                    SET community_report_id = ?
                    WHERE id = ?
                """,
                    (c_id, entity),
                )
        community_report = f"{mygo_graph.query(community)}\n\n"
        community_report_summary = cp.extract(community_report)
        with sqlite3.connect("db/mygo.db") as conn:
            cur = conn.cursor()
            cur.execute(
                """
                    INSERT INTO community_report (id, report)
                    VALUES(?, ?)
                """,
                (c_id, community_report_summary.model_dump_json()),
            )
        print(c_id)
