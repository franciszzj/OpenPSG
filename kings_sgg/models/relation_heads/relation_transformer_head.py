import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.builder import HEADS

from kings_sgg.datasets.coco_panoptic_relation import relation_description_dict
from kings_sgg.models.commons.bert_wrapper import BertWrapper
from kings_sgg.models.commons.clip_wrapper import CLIPWrapper
from kings_sgg.models.relation_heads.interactive_fusion_module import InteractiveFusionModule


@HEADS.register_module()
class RelationTransformerHead(BaseModule):
    def __init__(self,
                 pretrained_transformer_type='bert',
                 pretrained_transformer=None,
                 load_pretrained_weights=True,
                 use_adapter=False,
                 use_relation_prompts=False,
                 semantic_transformer_learnable=False,
                 semantic_transformer_type='bert',
                 semantic_transformer=None,
                 semantic_use_adapter=False,
                 semantic_type='relation_classes',
                 use_learnable_prompts=False,
                 learnable_prompts_size=16,
                 input_feature_size=256,
                 output_feature_size=768,
                 num_transformer_layer='all',
                 num_object_classes=133,
                 num_relation_classes=56,
                 cls_qk_size=512,
                 max_object_num=30,
                 embedding_add_cls=True,
                 merge_cls_type='add',
                 positional_encoding=None,
                 use_background_feature=False,
                 pred_type='attention',
                 graph_transformer_type=None,
                 loss_type='v1',
                 loss_weight=1.,
                 loss_alpha=None,):
        super().__init__()
        ##########
        # config #
        ##########
        self.pretrained_transformer_type = pretrained_transformer_type
        self.use_relation_prompts = use_relation_prompts
        self.semantic_transformer_learnable = semantic_transformer_learnable
        self.semantic_transformer = semantic_transformer
        self.semantic_text = relation_description_dict[semantic_type]
        self.use_learnable_prompts = use_learnable_prompts
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.num_transformer_layer = num_transformer_layer
        self.num_object_classes = num_object_classes
        if loss_type == 'v0_softmax':
            # add background, the last one is background
            num_relation_classes += 1
        self.num_relation_classes = num_relation_classes
        self.cls_qk_size = cls_qk_size
        self.max_object_num = max_object_num
        self.embedding_add_cls = embedding_add_cls
        self.merge_cls_type = merge_cls_type
        if positional_encoding is not None:
            self.positional_encoding_cfg = positional_encoding
            self.postional_encoding_layer = build_positional_encoding(
                positional_encoding)
        self.use_background_feature = use_background_feature
        self.pred_type = pred_type
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.loss_alpha = loss_alpha

        #########
        # model #
        #########
        # fc input
        if merge_cls_type == 'add':
            self.fc_input = nn.Sequential(
                nn.Linear(input_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
        elif merge_cls_type == 'cat':
            self.fc_input = nn.Sequential(
                nn.Linear(input_feature_size * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
        # fc output
        self.fc_output = nn.Sequential(
            nn.Linear(output_feature_size, output_feature_size),
            nn.LayerNorm(output_feature_size),
        )
        # transformer model
        if pretrained_transformer_type == 'bert':
            self.model = BertWrapper(pretrained_transformer, load_pretrained_weights,
                                     num_transformer_layer, use_adapter)
        elif pretrained_transformer_type == 'clip_v':
            self.model = CLIPWrapper(
                pretrained_transformer, use_adapter, model_keep='vision')
        elif pretrained_transformer_type == 'clip_t':
            self.model = CLIPWrapper(
                pretrained_transformer, use_adapter, model_keep='text')
        # relation semantic transformer model
        if use_relation_prompts:
            if semantic_transformer_type == 'bert':
                self.semantic_model = BertWrapper(
                    semantic_transformer, use_adapter=semantic_use_adapter,
                    use_learnable_prompts=use_learnable_prompts)
                self.semantic_feature_size = output_feature_size
            elif semantic_transformer_type == 'clip':
                self.semantic_model = CLIPWrapper(
                    semantic_transformer, semantic_use_adapter,
                    use_learnable_prompts=use_learnable_prompts, model_keep='text')
                self.semantic_feature_size = self.semantic_model.embed_dim
            if use_learnable_prompts:
                learnable_prompts_embedding = torch.zeros(
                    (num_relation_classes, learnable_prompts_size, self.semantic_feature_size))
                nn.init.normal_(learnable_prompts_embedding, std=0.02)
                self.learnable_prompts_embedding = nn.Parameter(
                    learnable_prompts_embedding)
            else:
                self.learnable_prompts_embedding = None
            if not self.semantic_transformer_learnable and not use_learnable_prompts:
                self.semantic_embedding = self.semantic_model.forward_texts(
                    self.semantic_text)
                del self.semantic_model
        # pred task head
        if not use_relation_prompts:
            self.cls_q = nn.Linear(output_feature_size,
                                   cls_qk_size * num_relation_classes)
            self.cls_k = nn.Linear(output_feature_size,
                                   cls_qk_size * num_relation_classes)
        else:
            self.fusion = InteractiveFusionModule(
                pred_type, output_feature_size, self.semantic_feature_size, cls_qk_size, graph_transformer_type)

        ########
        # loss #
        ########
        if loss_type == 'v0_softmax':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'v0_sigmoid':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type in ['v1', 'v1_no_bs_limit']:
            print('Use multilabel_categorical_crossentropy.')
        else:
            assert False, 'Please use support loss type.'

    def forward(self, inputs_embeds, attention_mask=None):
        # 1. fc input
        if inputs_embeds.shape[-1] != self.output_feature_size:
            encode_inputs_embeds = self.fc_input(inputs_embeds)
        else:
            encode_inputs_embeds = inputs_embeds
        # 2. transformer based model (subject-object pair)
        if self.pretrained_transformer_type == 'bert':
            position_ids = torch.ones([1, inputs_embeds.shape[1]]).to(
                inputs_embeds.device).to(torch.long)
            encode_embedding = self.model.forward_embeds(embeds=encode_inputs_embeds,
                                                         attention_mask=attention_mask,
                                                         position_ids=position_ids)
        elif self.pretrained_transformer_type == 'clip_v':
            if attention_mask is not None:
                attention_mask = torch.matmul(attention_mask.unsqueeze(2),
                                              attention_mask.unsqueeze(1)).unsqueeze(1)
            encode_embedding = self.model.forward_embeds_vision(embeds=encode_inputs_embeds,
                                                                attention_mask=attention_mask)
        elif self.pretrained_transformer_type == 'clip_t':
            if attention_mask is not None:
                attention_mask = torch.matmul(attention_mask.unsqueeze(2),
                                              attention_mask.unsqueeze(1)).unsqueeze(1)
            encode_embedding = self.model.forward_embeds_text(embeds=encode_inputs_embeds,
                                                              attention_mask=attention_mask)
        # 3. fc output
        encode_embedding = self.fc_output(encode_embedding)
        # 4. task head
        if not self.use_relation_prompts:
            batch_size, object_num, embedding_size = encode_embedding.shape
            q_embedding = self.cls_q(encode_embedding).reshape(
                [batch_size, object_num, self.num_relation_classes, self.cls_qk_size]).permute([0, 2, 1, 3])
            k_embedding = self.cls_k(encode_embedding).reshape(
                [batch_size, object_num, self.num_relation_classes, self.cls_qk_size]).permute([0, 2, 1, 3])
            if self.pred_type == 'attention':
                cls_pred = q_embedding @ torch.transpose(k_embedding, 2, 3) / self.cls_qk_size ** 0.5  # noqa
            elif self.pred_type == 'einsum':
                cls_pred = torch.einsum(
                    'nrsc,nroc->nrso', q_embedding, k_embedding)
            else:
                assert False, 'Not support pred_type: {}'.format(
                    self.pred_type)
        else:
            if self.semantic_transformer_learnable or self.use_learnable_prompts:
                if self.use_learnable_prompts:
                    learnable_prompts_embeds = self.learnable_prompts_embedding
                else:
                    learnable_prompts_embeds = None
                semantic_embedding = self.semantic_model.forward_texts(
                    self.semantic_text, learnable_embeds=learnable_prompts_embeds)
            else:
                semantic_embedding = self.semantic_embedding.detach().to(inputs_embeds.device)
            cls_pred = self.fusion(
                node_embed=encode_embedding, edge_embed=semantic_embedding)
        return cls_pred

    def loss(self, pred, target, mask_attention):
        """
            pred: [batch_size, 56, object_num, object_num]
            target: [batch_size, 56, object_num, object_num]
            mask_attention: [batch_size, 56, object_num, object_num]
        """

        losses = dict()
        batch_size, relation_num, object_num, object_num = pred.shape

        mask = torch.zeros_like(pred).to(pred.device)
        for idx in range(batch_size):
            n = torch.sum(mask_attention[idx]).to(torch.int)
            mask[idx, :, :n, :n] = 1
            if self.loss_type == 'v0_softmax':
                target[idx, :, n:, n:] = -100
        pred = pred * mask - 9999 * (1 - mask)

        if self.loss_type == 'v0_softmax':
            target = target[:, 0].to(torch.long)
            loss = self.loss_fn(pred, target)
        elif self.loss_type == 'v0_sigmoid':
            loss = self.loss_fn(pred, target)
        elif self.loss_type == 'v1':
            assert pred.shape[0] == 1 and target.shape[0] == 1
            input_tensor = pred.reshape([relation_num, -1])
            target_tensor = target.reshape([relation_num, -1])
            loss = self.multilabel_categorical_crossentropy(
                target_tensor, input_tensor)
            weight = (loss / loss.max()) ** self.loss_alpha
            loss = loss * weight
        elif self.loss_type == 'v1_no_bs_limit':
            input_tensor = torch.permute(pred, (1, 0, 2, 3))
            target_tensor = torch.permute(target, (1, 0, 2, 3))
            input_tensor = pred.reshape([relation_num, -1])
            target_tensor = target.reshape([relation_num, -1])
            loss = self.multilabel_categorical_crossentropy(
                target_tensor, input_tensor)
            weight = (loss / loss.max()) ** self.loss_alpha
            loss = loss * weight

        loss = loss.mean()
        losses['loss_relation'] = loss * self.loss_weight

        if self.loss_type != 'v0_softmax':
            # f1, p, r
            # [f1, precise, recall], [f1_mean, precise_mean,
            #                         recall_mean] = self.get_f1_p_r(pred, target, mask)
            # losses['relation.f1'] = f1
            # losses['relation.precise'] = precise
            # losses['relation.recall'] = recall
            # losses['relation.f1_mean'] = f1_mean
            # losses['relation.precise_mean'] = precise_mean
            # losses['relation.recall_mean'] = recall_mean

            # recall
            recall_20 = self.get_recall_N(pred, target, mask, object_num=20)
            # recall_50 = self.get_recall_N(pred, target, mask, object_num=50)
            # recall_100 = self.get_recall_N(pred, target, mask, object_num=100)
            losses['relation.recall@20'] = recall_20
            # losses['relation.recall@50'] = recall_50
            # losses['relation.recall@100'] = recall_100

        return losses

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def get_f1_p_r(self, y_pred, y_true, mask_attention, th=0):
        """
            y_pred: [batch_size, 56, object_num, object_num]
            y_true: [batch_size, 56, object_num, object_num]
            mask_attention: [batch_size, 56, object_num, object_num]
        """
        res = []

        y_pred[y_pred > th] = 1
        y_pred[y_pred < th] = 0

        n1 = y_pred * y_true * mask_attention
        n2 = y_pred * mask_attention
        n3 = y_true * mask_attention

        p = 100 * n1.sum(dim=[1, 2, 3]) / (1e-8 + n2.sum(dim=[1, 2, 3]))
        r = 100 * n1.sum(dim=[1, 2, 3]) / (1e-8 + n3.sum(dim=[1, 2, 3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([f1.mean(), p.mean(), r.mean()])

        mask_mean = y_true.sum(dim=[0, 2, 3]) > 0
        p = 100 * n1.sum(dim=[0, 2, 3]) / (1e-8 + n2.sum(dim=[0, 2, 3]))
        r = 100 * n1.sum(dim=[0, 2, 3]) / (1e-8 + n3.sum(dim=[0, 2, 3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([
            torch.sum(f1 * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(p * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(r * mask_mean) / (torch.sum(mask_mean) + 1e-8),
        ])

        return res

    def get_recall_N(self, y_pred, y_true, mask_attention, object_num=20):
        """
            y_pred: [batch_size, 56, object_num, object_num]
            y_true: [batch_size, 56, object_num, object_num]
            mask_attention: [batch_size, 56, object_num, object_num]
        """

        device = y_pred.device
        dtype = y_pred.dtype
        recall_list = []

        for idx in range(len(y_true)):
            sample_y_true = []
            sample_y_pred = []

            # find topk
            _, topk_indices = torch.topk(
                y_true[idx:idx+1].reshape([-1, ]), k=object_num)
            for index in topk_indices:
                # pred_cls = index // (y_true.shape[2] ** 2)
                pred_cls = torch.div(
                    index, y_true.shape[2] ** 2, rounding_mode='floor')
                index_subject_object = index % (y_true.shape[2] ** 2)
                # pred_subject = index_subject_object // y_true.shape[2]
                pred_subject = torch.div(
                    index_subject_object, y_true.shape[2], rounding_mode='floor')
                pred_object = index_subject_object % y_true.shape[2]
                if y_true[idx, pred_cls, pred_subject, pred_object] == 0:
                    continue
                sample_y_true.append([pred_subject, pred_object, pred_cls])

            # find topk
            _, topk_indices = torch.topk(
                y_pred[idx:idx+1].reshape([-1, ]), k=object_num)
            for index in topk_indices:
                # pred_cls = index // (y_pred.shape[2] ** 2)
                pred_cls = torch.div(
                    index, y_pred.shape[2] ** 2, rounding_mode='floor')
                index_subject_object = index % (y_pred.shape[2] ** 2)
                # pred_subject = index_subject_object // y_pred.shape[2]
                pred_subject = torch.div(
                    index_subject_object, y_pred.shape[2], rounding_mode='floor')
                pred_object = index_subject_object % y_pred.shape[2]
                sample_y_pred.append([pred_subject, pred_object, pred_cls])

            recall = len([x for x in sample_y_pred if x in sample_y_true]
                         ) / (len(sample_y_true) + 1e-8)
            recall_list.append(recall)

        mean_recall = torch.tensor(recall_list).to(device).mean() * 100
        return mean_recall
