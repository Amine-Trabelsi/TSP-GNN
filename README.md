## GraphSAGE For Solving the Traveling Salesman Problem (TSP)

how to use a graph neural network (GNN) architecture combining **GraphSAGE** layers to solve the Traveling Salesman Problem (TSP). The model is based on the approach described in the paper **"GatedGCN with GraphSAGE to Solve Traveling Salesman Problem" by Hua Yang**. We'll walk through the process of building and training this architecture on a dataset of graphs with varying sizes.

---

### **Introduction to the Problem**
The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where the goal is to find the shortest possible route that visits a set of cities exactly once and returns to the origin. Representing this problem as a graph:
- **Nodes**: Cities
- **Edges**: Paths between cities (with associated weights like distance or cost)

Graph neural networks are well-suited for TSP due to their ability to model relationships in graph-structured data.

---

### **Code**

https://github.com/Amine-Trabelsi/TSP-GNN

### **DataSet**

https://drive.google.com/drive/folders/10xyd_WaxhTcWhvRD57B53hibQ6zxH3l_?usp=sharing

---

## **Model Architecture**
This model, based on the paper **"GatedGCN with GraphSAGE to Solve the Traveling Salesman Problem" by Hua Yang**, is structured to efficiently process graph data and predict optimal TSP solutions. Below, each component of the model is described in detail, including the underlying calculations.

### **1. Input Layer**

The input to the model consists of:
- **Node Features (`x`)**: Represents the 2D coordinates of cities in the TSP graph. These are embedded into a higher-dimensional space (`d` dimensions) for richer representations.
- **Edge Features (`edge_attr`)**: Represents pairwise distances or other attributes between cities.

The transformation of node and edge features is as follows:
1. **Node Feature Embedding**:
   Each node is projected into a `d`-dimensional vector using a linear transformation:

```python
# init
self.node_embed = torch.nn.Linear(input_dim, output_dim)
# Forward
node_features = self.node_embed(x)  # L_i = A_1 x_i + b_1
```

2. **Edge Feature Embedding**:
   Edges are embedded into a \(d/2\)-dimensional space, incorporating both distance features and k-nearest neighbor indicators:

```python
# Build k-NN graph
edge_index = knn_graph(x, k=self.k, batch=batch, loop=True).to(device)

# Compute edge features
row, col = edge_index
edge_distances = torch.norm(x[row] - x[col], dim=1, keepdim=True).to(device)  # Distance between nodes
edge_indicator = torch.cat([
    (row == col).float().unsqueeze(1),  # Self-loop indicator
    torch.ones((edge_index.size(1), 1), device=device),  # k-NN indicator
    torch.zeros((edge_index.size(1), 1), device=device)  # Padding (can modify based on needs)
], dim=1)

# Embed edge features
edge_features = torch.cat([
    self.edge_embed_1(edge_distances),
    self.edge_embed_2(edge_indicator)
], dim=1)
```
---

### **2. GraphSAGE Layers (Message-Passing GCN Layer)**
GraphSAGE layers compute **node embeddings** by aggregating information from a node's neighbors. The update rule for the \(m\)-th layer is:


#### **`update_edge_features`**
**Purpose**: Update edge features based on the current node features and edge features.
- **Steps**:
  1. Extracts source (`row`) and target (`col`) node indices from `edge_index`.
  2. Updates edge features (`e_next`) using a weighted combination of:
     - Existing edge features transformed by `W3`.
     - Source node features transformed by `W4`.
     - Target node features transformed by `W5`.
  3. Applies batch normalization and a ReLU activation to stabilize and introduce non-linearity.
- **Key Role**: Refines edge features by combining information from both connected nodes and existing edge features.

---

#### **`message`**
**Purpose**: Generate messages to send to neighboring nodes during propagation.
- **Parameters**:
  - `x_j`: Features of neighboring nodes.
  - `edge_attr`: Features of edges connecting nodes.
- **Steps**:
  1. Computes attention scores (`alpha`) for each edge using a sigmoid function.
  2. Normalizes the attention scores across all edges connected to a node.
  3. Combines the neighbor features (`x_j`) with attention scores to generate messages.
- **Key Role**: Weighs the importance of neighboring nodes using attention mechanisms.

---

#### **`update`**
**Purpose**: Update the features of each node after aggregating messages.
- **Parameters**:
  - `aggr_out`: Aggregated messages from neighboring nodes.
  - `x`: Current node features.
- **Steps**:
  1. Combines the current node features transformed by `W1` with the aggregated messages (`aggr_out`).
  2. Applies batch normalization and ReLU activation to stabilize and refine the node updates.
- **Key Role**: Updates the node representations using both their current features and aggregated neighbor information.

---

#### **Summary of Data Flow in `CustomGCNLayer`**
1. **Input**: Node features (`x`), edge indices (`edge_index`), and edge features (`edge_attr`).
2. **Edge Feature Update**: Combines node and edge features to refine edge representations.
3. **Message Passing**:
   - Generates messages from neighboring nodes using attention-weighted features.
   - Aggregates messages using summation.
4. **Node Feature Update**: Combines current node features and aggregated messages to produce updated node representations.
5. **Output**: Updated node features and edge features.
---


### **3. Edge Prediction (MLP Classifier Layer)**
The MLPClassifierLayer is a key component for classifying edges in a graph, used to determine if an edge is part of a Traveling Salesman Problem (TSP) tour.

The final step predicts the probability of each edge being part of the TSP solution. For each edge \((i, j)\):
1. Concatenate the embeddings of the connected nodes:
2. Pass the concatenated embeddings through a Multi-Layer Perceptron (MLP):
Purpose:
- Outputs edge-level probabilities for the adjacency matrix, representing the likelihood of edges being part of the optimal TSP path.

```python
def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=5):
    """
    Initialize the MLP Classifier Layer.
    :param input_dim: Dimension of node features (h_i, h_j).
    :param hidden_dim: Hidden dimension for the MLP.
    :param output_dim: Number of classes (default: 2 for binary classification).
    :param num_layers: Number of fully connected layers in the MLP.
    """
    super(MLPClassifierLayer, self).__init__()
    
    layers = []
    layers.append(nn.Linear(2 * input_dim, hidden_dim))  # First layer
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))  # Hidden layers
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))  # Output layer

    self.mlp = nn.Sequential(*layers)
```

**Use Case**


Define the Layer:

```python
mlp_classifier = MLPClassifierLayer(input_dim=16, hidden_dim=256, output_dim=2, num_layers=5)
```

Pass Edge Features Through the MLP:

```python
logits = mlp_classifier(node_features, edge_index)
```

Interpret the Output:

Use softmax to compute probabilities:

```python
edge_probs = F.softmax(logits, dim=1)[:, 1]
```

Apply a threshold to classify edges:

```python
edge_predictions = edge_index[:, edge_probs > 0.5]
```

Use the predicted edges to construct a tour or visualize the graph.

---

## **Full Model Code**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GatedGraphConv

class TSPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TSPModel, self).__init__()
        
        # GraphSAGE Layers
        self.graphsage_layers = nn.ModuleList([
            SAGEConv(input_dim if i == 0 else hidden_dim, hidden_dim) 
            for i in range(num_layers)
        ])
        
        # GatedGCN Layer
        self.gated_gcn = GatedGraphConv(hidden_dim, num_layers)
        
        # MLP for edge predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # GraphSAGE: Node Embedding
        for layer in self.graphsage_layers:
            x = F.relu(layer(x, edge_index))
        
        # GatedGCN: Node and Edge Embedding
        x = self.gated_gcn(x, edge_index, edge_attr)
        
        # Edge Predictions
        edge_predictions = []
        for edge in edge_index.t():
            h_i, h_j = x[edge[0]], x[edge[1]]
            edge_predictions.append(self.mlp(torch.cat([h_i, h_j], dim=-1)))
        
        return torch.cat(edge_predictions, dim=0)
```


### **Advantages of This Approach**
1. **Scalability**: GraphSAGE and GatedGCN layers support variable-sized graphs, making this approach suitable for real-world TSP instances with diverse numbers of cities.
2. **Rich Representations**
3. **Flexibility**: Can handle additional node or edge features, such as traffic conditions or road types.

### References

* Michael Atkin. (2023). Tackling the Traveling Salesman Problem with Graph Neural Networks.
* Hua Yang. (2023). GatedGCN with GraphSage to Solve Traveling Salesman Problem.
