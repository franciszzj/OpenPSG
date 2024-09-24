import torch
import torch.nn as nn
import torch.nn.functional as F

from kings_sgg.models.commons.graph_transformer import GraphTransformer


class InteractiveFusionModule(nn.Module):
    def __init__(self, pred_type, object_input_size, relation_input_size, output_size, graph_transformer_type=None):
        super().__init__()
        self.pred_type = pred_type
        self.graph_transformer_type = graph_transformer_type
        if self.graph_transformer_type is not None:
            self.fc_node = nn.Linear(object_input_size, output_size)
            self.fc_edge = nn.Linear(relation_input_size, output_size)
            self.graph_transformer = GraphTransformer(output_size, num_layer=3)
            self.cls_s = MLP(output_size, output_size*2, output_size)
            self.cls_o = MLP(output_size, output_size*2, output_size)
            self.cls_r = MLP(output_size, output_size*2, output_size)
        else:
            self.cls_s = nn.Linear(object_input_size, output_size)
            self.cls_o = nn.Linear(object_input_size, output_size)
            self.cls_r = nn.Linear(relation_input_size, output_size)
        if self.pred_type == 'attention':
            self.fc_q = MLP(output_size, output_size*2, output_size)
            self.fc_k = MLP(output_size, output_size*2, output_size)

    def forward(self, node_embed, edge_embed):
        if self.graph_transformer_type is not None:
            node_embed = self.fc_node(node_embed)
            edge_embed = self.fc_edge(edge_embed)
            node_embed, edge_embed = self.graph_transformer(
                node_embed, edge_embed)
        sub_embed = self.cls_s(node_embed)
        obj_embed = self.cls_o(node_embed)
        rel_embed = self.cls_r(edge_embed)
        if self.pred_type == 'attention':
            so_embed = torch.einsum('nsc,noc->nsoc', sub_embed, obj_embed)
            batch_size, sub_num, obj_num, feature_size = so_embed.shape
            rel_num, _ = rel_embed.shape
            so_embed = so_embed.reshape(
                (batch_size, sub_num*obj_num, feature_size))
            r_embed = rel_embed.unsqueeze(0).repeat([batch_size, 1, 1])
            so_embed = self.fc_q(so_embed)
            r_embed = self.fc_k(r_embed)
            pred = so_embed @ r_embed.transpose(1, 2) / feature_size ** 0.5
            pred = pred.reshape((batch_size, sub_num, obj_num, rel_num))
            pred = torch.permute(pred, (0, 3, 1, 2))
        elif self.pred_type == 'einsum':
            pred = torch.einsum(
                'nsc,noc,rc->nrso', sub_embed, obj_embed, rel_embed)
        elif self.pred_type == 'einsum_v1':
            pred_tmp = torch.einsum(
                'nsc,noc->nsoc', sub_embed, obj_embed)
            pred = torch.einsum(
                'nsoc,rc->nrso', pred_tmp, rel_embed)
        elif self.pred_type == 'einsum_v2':
            pred_score = torch.einsum(
                'nsc,noc->nso', sub_embed, obj_embed).sigmoid()
            pred_tmp = torch.einsum(
                'nsc,noc->nsoc', sub_embed, obj_embed)
            pred = torch.einsum(
                'nsoc,rc->nrso', pred_tmp, rel_embed) * pred_score.unsqueeze(1)
        else:
            assert False, 'Not support pred_type: {}'.format(
                self.pred_type)

        return pred


class MLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)
        self.act = nn.LayerNorm(intermediate_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
