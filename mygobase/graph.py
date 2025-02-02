import networkx as nx

from mygobase.entity_embedd import load_ner
from mygobase.ner import NER


class MyGOKnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_entity_and_relation(self, ner: NER):
        entites = ner.entities
        for e in entites:
            self.graph.add_node(e.name, type=e.type)
        relations = ner.relations
        for r in relations:
            self.graph.add_edge(r.subject, r.object, relation=r.relation)

    def query(self, entity: str):
        relevant_info = []
        for _, dst, data in self.graph.edges(entity, data=True):
            relevant_info.append(f"{entity} -> {data['relation']} -> {dst}")
        if relevant_info:
            return "相關信息：\n" + "\n".join(relevant_info)
        else:
            return "未找到相關信息。"


if __name__ == "__main__":
    ners = load_ner()
    print(f"#ner:{len(ners)}")
    mygo_graph = MyGOKnowledgeGraph()
    for ner in ners:
        mygo_graph.add_entity_and_relation(ner)
    result = mygo_graph.query("燈")
    print(result)
