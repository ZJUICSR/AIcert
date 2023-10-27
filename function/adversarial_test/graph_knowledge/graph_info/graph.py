import networkx as nx
import os
import json


NODE_FILE = os.path.join(os.path.dirname(__file__), 'node.json')
RELATION_FILE = os.path.join(os.path.dirname(__file__), 'relation.json')


class AttackKnowledge(object):
    def __init__(self,
                 node_file=NODE_FILE,
                 relation_file=RELATION_FILE):
        super(AttackKnowledge, self).__init__()
        self.node_file = node_file
        self.relation_file = relation_file
        self.graph = self.build_graph()

    @staticmethod
    def save_json_info(filename, info):
        with open(filename, 'a', encoding='utf-8') as file_obj:
            json.dump(info, file_obj)
            file_obj.write("\n")
        return

    @staticmethod
    def load_json(file_path) -> list:
        """
        将json文件转换为对象列表
        :param file_path:
        :return:
        """
        info_list = list()
        with open(file_path, 'r', encoding='utf-8') as file:
            info_list = json.load(file)

        return info_list

    def build_graph(self)->nx.DiGraph:
        G = nx.DiGraph()
        nodes = self.load_json(self.node_file)
        for node in nodes:
            G.add_node(node['name'], desc=node['desc'], paper=node['paper'], method_type=node['method_type'])

        relations = self.load_json(self.relation_file)
        for relation in relations:
            G.add_edge(relation['source'], relation['dest'], desc=relation['desc'])

        return G

    @staticmethod
    def get_edges_list(edges, node):
        return {d for (v, d) in edges if v == node}

    def recom(self, attack_mode: str, attack_type: str, data_type: str, defend_algorithm: str):
        edges = self.graph.edges()
        attack_modes_recom = self.get_edges_list(edges, attack_mode)
        attack_type_recom = self.get_edges_list(edges, attack_type)
        data_type_recom = self.get_edges_list(edges, data_type)
        defend_algorithm_recom = self.get_edges_list(edges, defend_algorithm)

        recom_algorithms = attack_modes_recom & attack_type_recom & data_type_recom

        if len(defend_algorithm_recom) != 0:
            recom_algorithms -= defend_algorithm_recom


        return list(recom_algorithms)


if __name__ == '__main__':
    knowledge = AttackKnowledge()
    result = knowledge.recom('白盒', '逃逸攻击', '图片', '')
    print(f'result={result}')


