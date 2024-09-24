import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformer(nn.Module):
    def __init__(self, feature_size, num_layer=3):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_size, nhead=8, batch_first=True)
        self.edge2node_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layer)
        self.node2edge_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layer)

    def forward(self, node_embed, edge_embed):
        b, l, c = node_embed.shape
        edge_embed = edge_embed.repeat(b, 1, 1)
        node_embed = self.edge2node_decoder(node_embed, edge_embed)
        edge_embed = self.node2edge_decoder(edge_embed, node_embed)
        edge_embed = edge_embed.mean(dim=0)
        return node_embed, edge_embed
