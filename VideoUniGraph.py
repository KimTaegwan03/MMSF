# Graph Constructor
from torch_geometric.data import Data
import torch_geometric.nn as geonn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# U = # of Utterance = # of Node
# D = Vector Dimension

# Input (n-modal features + speaker list)
# Text Features: [U, D]
# Image Features: [U, D]
# Audio Features: [U, D]
# Posual Features: [U, D]
# Facial Features: [U, D]
# Speaker Map: [U, 1]
# K(window size): int

# Output (node:[U+1, D], edge:[2, U+1])

class MoE(nn.Module):
    """Mixture of Experts module for cross-domain and cross-modality alignment"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        num_selected_experts: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get expert weights
        weights = self.gate(x)
        
        # Select top-k experts
        top_weights, top_indices = torch.topk(weights, self.num_selected_experts, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Combine expert outputs
        selected_outputs = torch.gather(
            expert_outputs,
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.size(-1))
        )

        output = torch.sum(selected_outputs * top_weights.unsqueeze(-1), dim=2)
        
        return output
    
class VolatilityDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
class SPDDecoder(nn.Module):
    """Shortest Path Distance decoder"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_i, x_j], dim=-1)
        return self.decoder(x)
    

class VideoUniGraph(nn.Module):
    def __init__(self,
        input_dims: Dict[str, int],  # Dictionary of input dimensions for each modality
        hidden_dim: int = 768,
        num_experts: int = 8,
        num_selected_experts: int = 2,
        num_layers: int = 3,
        feat_drop_rate: float = 0.1,
        edge_mask_rate: float = 0.1,
        gamma: float = 2.0,
        lambda_spd: float = 0.5,
        edge_window: int = 1,
        device:str = 'cuda'
    ):
        super(VideoUniGraph,self).__init__()
        self.hidden_dim = hidden_dim
        self.feat_drop_rate = feat_drop_rate
        self.edge_mask_rate = edge_mask_rate
        self.gamma = gamma
        self.lambda_spd = lambda_spd
        self.modals = input_dims.keys()
        self.K = edge_window
        self.device = device
        
        self.modal_projector = nn.ModuleDict([[
            modal, nn.Linear(input_dims[modal], hidden_dim)
        ] for modal in self.modals])
        
        # Mixture of Experts
        self.moe = MoE(hidden_dim, hidden_dim, num_experts, num_selected_experts)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            geonn.GATConv(hidden_dim, hidden_dim, heads=4),
            geonn.GATConv(hidden_dim*4, hidden_dim, heads=4),
            geonn.GATConv(hidden_dim*4, hidden_dim, heads=4)
            # for _ in range(num_layers-1)
        ])
        
        self.decoder = nn.ModuleDict([[
            modal, VolatilityDecoder(hidden_dim*4)  # Assuming 4 heads in GATConv
        ] for modal in self.modals])

        # SPD decoder
        self.spd_decoder = SPDDecoder(hidden_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(hidden_dim,dtype=torch.float32)).to(self.device)
        
        # Conv Vertex
        self.x_hie_conv_token = nn.Parameter(torch.randn(hidden_dim,dtype=torch.float32)).to(self.device)

        # Speaker Vertex
        self.x_hie_speaker_token = nn.Parameter(torch.randn(hidden_dim,dtype=torch.float32)).to(self.device)

    def construct_conversation_graph(
        self,
        x: torch.Tensor,
        K: int
    ):
        edge = []
        len_spmap = x.shape[0]
        
        # Sequencial Edge (Window Size = K)
        for i in range(len_spmap):
            for j in range(i-K,i-K+1):
                if j > 0 and j < len_spmap:
                    edge.append([i,j])
                    
        # Global Edge
        for i in range(len_spmap):
            edge.append([len_spmap,i])
            edge.append([i,len_spmap])
            
        edge_t = torch.Tensor(np.transpose(edge)).to(int).to(self.device)
        
        vertex = torch.concat([x,self.global_vertex])
        
        graph_data = Data(vertex, edge_t)
        
        return graph_data

    def _mask_features(
        self,
        features: torch.Tensor,
        mask_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask node features"""
        num_nodes = features.size(0)
        valid_nodes = num_nodes - 1
        num_masked = int(valid_nodes  * mask_rate)
        
        # Create mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        indices = torch.randperm(valid_nodes)[:num_masked]  # indices in range [0, num_nodes - 2]
        mask[indices] = True
        
        # Apply mask
        masked_features = features.clone()
        masked_features[mask] = self.mask_token
        
        return masked_features, mask
    
    def _compute_spd_loss(
        self,
        embeddings: torch.Tensor,
        spd_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute shortest path distance loss"""
        num_nodes = embeddings.size(0)
        spd_pred = torch.zeros_like(spd_matrix)
        
        # Compute predicted SPD for all node pairs
        for i in range(num_nodes):
            for j in range(num_nodes):
                spd_pred[i, j] = self.spd_decoder(embeddings[i], embeddings[j])
                
        return F.mse_loss(spd_pred, spd_matrix)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        graph: Data,  # Graph data structure
        # spmap: List,
        spd_matrix: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        proj = {}
        
        for modal, f in features.items():
            proj[modal] = self.modal_projector[modal](f)
        
        # Average features across modalities
        x = torch.stack(list(proj.values())).mean(dim=0)
        
        # Mask features
        masked_x, mask = self._mask_features(x, self.feat_drop_rate)
        
        # Apply MoE
        aligned_x = self.moe(masked_x)
        
        # graph = self.construct_conversation_graph(aligned_x, self.K)

        # Apply GNN layers
        x_hie_conv = self.x_hie_conv_token.unsqueeze(0).repeat(graph.x_hie_conv.size(0), 1)      # [hie_conv_count, 768]
        x_hie_speaker = self.x_hie_speaker_token.unsqueeze(0).repeat(graph.x_hie_speaker.size(0), 1)  # [hie_speaker_count, 768]

        h = torch.concat([aligned_x,x_hie_conv.unsqueeze(1),x_hie_speaker.unsqueeze(1)], dim=0).squeeze(1)

        print("Initial h shape:", h.shape)

        for layer in self.gnn_layers:
            h = layer(h, graph.edge_index)
        
        print("After GNN layers, h shape:", h.shape)

        res = {
                "conversation_embeddings": h[:aligned_x.size(0),:],  # Exclude hierarchical nodes
                "global_embeddings": h[graph.global_index[0]:graph.global_index[0]+1,:], # Global node embedding
                "speaker_embeddings": h[graph.speaker_index,:]
            }
            
        if return_embeddings:
            return res
            
        # Reconstruct features for each domain
        reconstruction_loss = 0
        for domain, decoder in self.decoder.items():
            reconstructed = decoder(h[:aligned_x.size(0),:][mask])
            original = features[domain][mask]
            similarity = F.cosine_similarity(reconstructed, original, dim=-1)
            reconstruction_loss += (1 - similarity).pow(self.gamma).mean()
            
        # Compute SPD loss if provided
        spd_loss = 0
        if spd_matrix is not None:
            spd_loss = self._compute_spd_loss(h, spd_matrix)
            
        # Combine losses
        total_loss = reconstruction_loss + self.lambda_spd * spd_loss
        
        return total_loss, res


if __name__ == "__main__":
    model = VideoUniGraph(
        input_dims={
            "text": 768,
            "video": 768,
            "audio": 768
        },
        device='cpu'
    )


    graph = torch.load('data/graph/trump_single.pt',map_location=torch.device('cpu'))

    # Print datatypes and shapes of graph features
    print("x_text dtype:", graph.x_text.dtype, "\tshape:", graph.x_text.shape)
    print("x_video dtype:", graph.x_video.dtype, "\tshape:", graph.x_video.shape)
    print("x_audio dtype:", graph.x_audio.dtype, "\tshape:", graph.x_audio.shape)

    pred = model.forward(
        features={
            "text": graph.x_text,
            "video": graph.x_video,
            "audio": graph.x_audio.to(torch.float32)
        },
        graph=graph,
        return_embeddings=False
    )

    print(pred)