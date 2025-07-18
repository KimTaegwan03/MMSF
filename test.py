from pyvis.network import Network
import networkx as nx
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BRANCH_FACTOR = 2 # 아직 2만 됨

def build_tree_edge_index(num_nodes, branch_factor,level_edge=True):
    if branch_factor < 1:
        raise ValueError("branch_factor는 1 이상이어야 합니다.")
    if num_nodes < 2:
        raise ValueError("노드 수는 최소 2개 이상이어야 트리 구조가 가능합니다.")

    # 1. 트리 depth 계산
    depth = 1
    total = 1
    prev = 1
    while total < num_nodes:
        total += (prev-1) * (branch_factor-1) + branch_factor
        depth += 1

    print("Depth:", depth, "Total Nodes:", total)

    # 실제 생성 가능한 노드 수를 초과한 경우, 마지막 층 일부는 잘릴 수 있음
    level_nodes = [[0]]
    node_id = 1
    previous_count = 1
    for d in range(1,depth):
        count = (previous_count-1) * (branch_factor-1) + branch_factor
        previous_count = count
        layer = []
        for _ in range(count):
            if node_id >= num_nodes:
                break
            layer.append(node_id)
            node_id += 1
        level_nodes.append(layer)
        if node_id >= num_nodes:
            break

    # 2. edge 구성 (상위 노드 → 하위 노드)
    edges = []
    for l in range(len(level_nodes) - 1):
        parents = level_nodes[l]
        children = level_nodes[l + 1]
        for i, p in enumerate(parents):
            if i != 0 and level_edge:  # 첫 번째 부모 노드는 제외
                edges.append((p, parents[i - 1]))
                edges.append((parents[i - 1], p))
            for j in range(branch_factor):
                c_index = i + j
                if c_index < len(children):
                    edges.append((p, children[c_index]))
                    edges.append((children[c_index], p))

    # 마지막 층의 노드가 부족한 경우, 마지막 층의 노드끼리 연결
    if level_edge and len(level_nodes[-1]) > 1:
        last_layer = level_nodes[-1]
        for i in range(len(last_layer) - 1):
            edges.append((last_layer[i], last_layer[i + 1]))
            edges.append((last_layer[i + 1], last_layer[i]))

    # 3. edge_index 반환
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

speaker_num = [0,1,0,2,1,2,1]

node_count = len(speaker_num)  # 노드 개수

edge_index = None

speaker_color = ['blue', 'green', 'orange', 'skyblue', 'cyan']  # 화자별 색상 (최대 5명)

node_color = []

# 대화 그래프에 Sequential 엣지 추가
for idx in range(node_count):
    node_color.append('red')
    if idx == 0:
        continue

    # 이전 노드와 현재 노드 간의 엣지 추가
    if edge_index is not None:
        edge_index = torch.cat([edge_index, torch.tensor([[idx - 1, idx], [idx, idx - 1]], dtype=torch.long)],dim=1)
    else:
        edge_index = torch.tensor([[idx - 1, idx], [idx, idx - 1]], dtype=torch.long)

# 대화 그래프에 Hierarchical-conversation 노드 임베딩 초기화
hie_conv_count = (node_count - 1) * node_count // 2
x_hie_conv = torch.zeros((hie_conv_count, 1), dtype=torch.float)  # Hierarchical-conversation 노드 임베딩 초기화 (추후 nn.Parameter로 교체 가능)

# 대화 그래프에 Hierarchical-conversation 엣지 추가
edge_index_hie_conv = build_tree_edge_index(hie_conv_count, branch_factor=BRANCH_FACTOR)

# conversation_graph의 edge_index와 edge_index_hie_conv를 연결
edge_index_hie_conv = edge_index_hie_conv + node_count # offset  # Hierarchical-conversation 엣지 인덱스에 오프셋 추가
edge_index = torch.cat([edge_index, edge_index_hie_conv], dim=1)

for i in range(hie_conv_count):
    node_color.append('purple')

for i in range(node_count):
    if i == 0:
        edge_index = torch.cat([edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - 1 - i]], dtype=torch.long)], dim=1)
        edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count - 1 - i], [node_count - i - 1]], dtype=torch.long)], dim=1)
    elif i == node_count-1:
        edge_index = torch.cat([edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - i]], dtype=torch.long)], dim=1)
        edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count - i], [node_count - i - 1]], dtype=torch.long)], dim=1)
    else:
        edge_index = torch.cat([edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - i]], dtype=torch.long)], dim=1)
        edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count - i], [node_count - i - 1]], dtype=torch.long)], dim=1)

        edge_index = torch.cat([edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - 1 - i]], dtype=torch.long)], dim=1)
        edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count - 1 - i], [node_count - i - 1]], dtype=torch.long)], dim=1)

# 대화 그래프에 Hierarchical-speaker 노드 임베딩 초기화
# 각 화자마다 노드 개수를 구하고 Hierarchical-conversation과 유사한 방식으로 노드 임베딩을 초기화
prev_speaker_num = 0
unique_speakers = set(speaker_num)

for speaker in unique_speakers:
    speaker_indices = [i for i, s in enumerate(speaker_num) if s == speaker]
    speaker_node_count = len(speaker_indices)
    if speaker_node_count < 2:
        continue
    
    # Hierarchical-speaker 노드 임베딩 초기화
    hie_speaker_count = (speaker_node_count - 1) * speaker_node_count // 2
    x_hie_speaker = torch.zeros((hie_speaker_count, 1), dtype=torch.float)

    for i in range(hie_speaker_count):
        node_color.append(speaker_color[speaker_indices[0]])

    if hie_speaker_count != 1:

        # Hierarchical-speaker 엣지 추가
        edge_index_hie_speaker = build_tree_edge_index(hie_speaker_count, branch_factor=BRANCH_FACTOR)

        # conversation_graph의 edge_index와 edge_index_hie_speaker를 연결
        edge_index_hie_speaker = edge_index_hie_speaker + node_count + hie_conv_count + prev_speaker_num # offset  # Hierarchical-speaker 엣지 인덱스에 오프셋 추가
        edge_index = torch.cat([edge_index, edge_index_hie_speaker], dim=1)

    for i in range(len(speaker_indices)): # s는 현재 화자의 대화 노드 인덱스
        s = speaker_indices[-1-i]
        if i == 0:
            edge_index = torch.cat([edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i]], dtype=torch.long)], dim=1)
            edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i], [s]], dtype=torch.long)], dim=1)
        if i == len(speaker_indices)-1:
            edge_index = torch.cat([edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num  - i]], dtype=torch.long)], dim=1)
            edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - i], [s]], dtype=torch.long)], dim=1)
        if i != 0 and i != len(speaker_indices)-1:
            edge_index = torch.cat([edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - i]], dtype=torch.long)], dim=1)
            edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - i], [s]], dtype=torch.long)], dim=1)
            edge_index = torch.cat([edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i]], dtype=torch.long)], dim=1)
            edge_index = torch.cat([edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i], [s]], dtype=torch.long)], dim=1)

    prev_speaker_num += hie_speaker_count

# 그래프 그리기

G = nx.DiGraph()

for edge in edge_index.t().tolist():
    G.add_edge(edge[0], edge[1])

# pos = nx.spring_layout(G)  # 노드 위치 설정
net = Network(notebook=False, height='900px', width='100%')
net.from_nx(G)

for node in net.nodes:
    node["label"] = str(node["id"])  # id를 문자열로 label로 사용
    node["color"] = node_color[node["id"]]  # 기존 색상 그대로
net.show('graph.html',notebook=False)