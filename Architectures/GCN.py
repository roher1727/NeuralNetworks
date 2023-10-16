import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load the CORA dataset (a benchmark dataset for node classification in a graph)
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

# Instantiate the model and optimizer
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    # Use the first graph in the dataset as data
    data = dataset[0]
    # Get the output from the model
    out = model(data)
    # Compute the loss (only for labeled nodes, to simulate semi-supervised setting)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # Backpropagate
    loss.backward()
    optimizer.step()
    return loss.item()

# Train for 200 epochs
for epoch in range(200):
    loss = train()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
