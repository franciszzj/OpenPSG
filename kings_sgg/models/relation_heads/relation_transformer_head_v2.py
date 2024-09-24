import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.builder import HEADS

from kings_sgg.models.commons.bert_wrapper import BertWrapper


@HEADS.register_module()
class RelationTransformerHeadV2(BaseModule):
    def __init__(self,
                 pretrained_transformer=None,
                 load_pretrained_weights=True,
                 use_adapter=False,
                 input_feature_size=256,
                 output_feature_size=768,
                 num_transformer_layer='all',
                 num_object_classes=133,
                 num_relation_classes=56,
                 text_embedding_size=1536,
                 max_object_num=30,
                 use_object_vision_only=True,
                 use_pair_vision_only=False,
                 use_pair_text_vision_cross=False,
                 use_pair_vision_text_cross=False,
                 use_triplet_vision_text_cross=False,
                 use_moe=False,
                 moe_weight_type='v1',
                 embedding_add_cls=True,
                 merge_cls_type='add',
                 positional_encoding=None,
                 use_background_feature=False,
                 loss_type='v1',
                 loss_weight=1.,
                 loss_alpha=None,):
        super().__init__()
        ##########
        # config #
        ##########
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.num_transformer_layer = num_transformer_layer
        self.num_object_classes = num_object_classes
        if loss_type == 'v0_softmax':
            # add background, the last one is background
            num_relation_classes += 1
        self.num_relation_classes = num_relation_classes
        self.max_object_num = max_object_num
        self.use_object_vision_only = use_object_vision_only
        self.use_pair_vision_only = use_pair_vision_only
        self.use_pair_text_vision_cross = use_pair_text_vision_cross
        self.use_pair_vision_text_cross = use_pair_vision_text_cross
        self.use_triplet_vision_text_cross = use_triplet_vision_text_cross
        self.use_moe = use_moe
        self.moe_weight_type = moe_weight_type
        self.embedding_add_cls = embedding_add_cls
        self.merge_cls_type = merge_cls_type
        if positional_encoding is not None:
            self.positional_encoding_cfg = positional_encoding
            self.postional_encoding_layer = build_positional_encoding(
                positional_encoding)
        self.use_background_feature = use_background_feature
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.loss_alpha = loss_alpha

        #########
        # model #
        #########
        if self.use_object_vision_only:
            self.object_level_model = BertWrapper(pretrained_transformer, load_pretrained_weights,
                                                  num_transformer_layer, use_adapter)
        if self.use_pair_vision_only or self.use_pair_text_vision_cross or self.use_pair_vision_text_cross:
            self.sub_obj_pair_level_model = BertWrapper(pretrained_transformer, load_pretrained_weights,
                                                        num_transformer_layer, use_adapter, add_cross_attention=True)

        if self.use_object_vision_only:
            # object level, vision_only
            self.fc_object_vision_only_input = nn.Sequential(
                nn.Linear(input_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_object_vision_only_output = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.object_vision_only_sub_pred = nn.Linear(output_feature_size,
                                                         output_feature_size * num_relation_classes)
            self.object_vision_only_obj_pred = nn.Linear(output_feature_size,
                                                         output_feature_size * num_relation_classes)
        if self.use_pair_vision_only:
            # sub_obj_pair level, vision only
            self.fc_pair_vision_only_input = nn.Sequential(
                nn.Linear(input_feature_size * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_pair_vision_only_output = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.pair_vision_only_pred = nn.Linear(output_feature_size,
                                                   num_relation_classes)
        if self.use_pair_text_vision_cross:
            # sub_obj_pair level, text vision cross
            self.fc_pair_text_vision_cross_input = nn.Sequential(
                nn.Linear(text_embedding_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_pair_vision_prompt_input = nn.Sequential(
                nn.Linear(input_feature_size * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_pair_text_vision_cross_output = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.pair_text_vision_cross_pred = nn.Linear(output_feature_size,
                                                         num_relation_classes)
        if self.use_pair_vision_text_cross:
            # sub_obj_pair level, vision text cross
            self.fc_pair_vision_text_cross_input = nn.Sequential(
                nn.Linear(input_feature_size * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_pair_text_prompt_input = nn.Sequential(
                nn.Linear(text_embedding_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_pair_vision_text_cross_output = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.pair_vision_text_cross_pred = nn.Linear(output_feature_size,
                                                         num_relation_classes)

        if self.use_triplet_vision_text_cross:
            # sub_obj_rel_triplet level, vision text cross
            self.fc_triplet_vision_text_cross_input = nn.Sequential(
                nn.Linear(input_feature_size * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_triplet_text_prompt_input = nn.Sequential(
                nn.Linear(text_embedding_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.fc_triplet_vision_text_cross_output = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.triplet_vision_text_cross_pred = nn.ModuleList(
                [nn.Linear(output_feature_size, 1) for _ in range(self.num_relation_classes)])

        if self.use_moe:
            expert_num = 0
            expert_num += 1 if self.use_object_vision_only else 0
            expert_num += 1 if self.use_pair_vision_only else 0
            expert_num += 1 if self.use_pair_text_vision_cross else 0
            expert_num += 1 if self.use_pair_vision_text_cross else 0
            expert_num += 1 if self.use_triplet_vision_text_cross else 0
            if self.moe_weight_type == 'v1':
                moe_output_feature_size = expert_num
            elif self.moe_weight_type == 'v2':
                moe_output_feature_size = num_relation_classes * expert_num
            self.moe_input_pair_vision = nn.Sequential(
                nn.Linear(input_feature_size * 2, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.moe_input_pair_text = nn.Sequential(
                nn.Linear(text_embedding_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.moe_input_triplet_text = nn.Sequential(
                nn.Linear(text_embedding_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.moe = nn.Sequential(
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),
                nn.ReLU(inplace=True),
                nn.Linear(output_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),
                nn.ReLU(inplace=True),
                nn.Linear(output_feature_size, moe_output_feature_size),)

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

    def forward(self, inputs):
        # 0. parse inputs
        object_embedding = inputs['object_embedding']
        batch_size, object_num, _ = object_embedding.shape
        sub_obj_pair_embedding = inputs['sub_obj_pair_embedding']
        sub_obj_pair_str_list = inputs['sub_obj_pair_str_list']
        sub_obj_pair_text_embedding = inputs['sub_obj_pair_text_embedding']
        sub_obj_pair_mask_attention = inputs['sub_obj_pair_mask_attention']
        sub_obj_rel_triplet_text_embedding = inputs['sub_obj_rel_triplet_text_embedding']

        if self.use_moe:
            moe_pred_list = []

        if self.use_object_vision_only:
            # 1. object level
            position_ids = torch.zeros(
                object_embedding.size()[:-1], dtype=torch.long, device=object_embedding.device)
            object_embedding = self.fc_object_vision_only_input(
                object_embedding)
            object_embedding = self.object_level_model.forward_embeds(
                object_embedding,
                position_ids=position_ids)
            object_embedding = self.fc_object_vision_only_output(
                object_embedding)
            sub_embedding = self.object_vision_only_sub_pred(object_embedding).reshape(
                [batch_size, object_num, self.num_relation_classes, self.output_feature_size]).permute([0, 2, 1, 3])
            obj_embedding = self.object_vision_only_obj_pred(object_embedding).reshape(
                [batch_size, object_num, self.num_relation_classes, self.output_feature_size]).permute([0, 2, 1, 3])
            object_vision_only_pred = torch.einsum(
                'nrsc,nroc->nrso', sub_embedding, obj_embedding)
            if self.use_moe:
                moe_pred_list.append(object_vision_only_pred)
        else:
            object_vision_only_pred = None

        if self.use_pair_vision_only:
            # 2. sub_obj_pair level, vision only
            position_ids = torch.zeros(
                sub_obj_pair_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_embedding.device)
            token_type_ids = torch.zeros(
                sub_obj_pair_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_embedding.device)
            sub_obj_pair_embedding_2 = self.fc_pair_vision_only_input(
                sub_obj_pair_embedding)
            vision_only_output = self.sub_obj_pair_level_model(
                inputs_embeds=sub_obj_pair_embedding_2,
                attention_mask=sub_obj_pair_mask_attention,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                return_dict=True,)
            vision_only_embedding = vision_only_output.last_hidden_state
            vision_only_embedding = self.fc_pair_vision_only_output(
                vision_only_embedding)
            pair_vision_only_pred = self.pair_vision_only_pred(
                vision_only_embedding).permute([0, 2, 1]).reshape(
                [-1, self.num_relation_classes, object_num, object_num])
            if self.use_moe:
                moe_pred_list.append(pair_vision_only_pred)
        else:
            pair_vision_only_pred = None

        if self.use_pair_text_vision_cross:
            # 3. sub_obj_pair level, text vision cross
            position_ids = torch.zeros(
                sub_obj_pair_text_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_text_embedding.device)
            token_type_ids = torch.zeros(
                sub_obj_pair_text_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_text_embedding.device)
            sub_obj_pair_text_embedding_3 = self.fc_pair_text_vision_cross_input(
                sub_obj_pair_text_embedding)
            sub_obj_pair_embedding_3 = self.fc_pair_vision_prompt_input(
                sub_obj_pair_embedding)
            text_vision_cross_output = self.sub_obj_pair_level_model(
                inputs_embeds=sub_obj_pair_text_embedding_3,
                encoder_hidden_states=sub_obj_pair_embedding_3,
                attention_mask=sub_obj_pair_mask_attention,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                return_dict=True,)
            text_vision_cross_embedding = text_vision_cross_output.last_hidden_state
            text_vision_cross_embedding = self.fc_pair_text_vision_cross_output(
                text_vision_cross_embedding)
            pair_text_vision_cross_pred = self.pair_text_vision_cross_pred(
                text_vision_cross_embedding).permute([0, 2, 1]).reshape(
                [-1, self.num_relation_classes, object_num, object_num])
            if self.use_moe:
                moe_pred_list.append(pair_text_vision_cross_pred)
        else:
            pair_text_vision_cross_pred = None

        if self.use_pair_vision_text_cross:
            # 4. sub_obj_pair_level, vision text cross
            position_ids = torch.zeros(
                sub_obj_pair_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_embedding.device)
            token_type_ids = torch.zeros(
                sub_obj_pair_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_embedding.device)
            sub_obj_pair_embedding_4 = self.fc_pair_vision_text_cross_input(
                sub_obj_pair_embedding)
            sub_obj_pair_text_embedding_4 = self.fc_pair_text_prompt_input(
                sub_obj_pair_text_embedding)
            vision_text_cross_output = self.sub_obj_pair_level_model(
                inputs_embeds=sub_obj_pair_embedding_4,
                encoder_hidden_states=sub_obj_pair_text_embedding_4,
                attention_mask=sub_obj_pair_mask_attention,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                return_dict=True,)
            vision_text_cross_embedding = vision_text_cross_output.last_hidden_state
            vision_text_cross_embedding = self.fc_pair_vision_text_cross_output(
                vision_text_cross_embedding)
            pair_vision_text_cross_pred = self.pair_vision_text_cross_pred(
                vision_text_cross_embedding).permute([0, 2, 1]).reshape(
                [-1, self.num_relation_classes, object_num, object_num])
            if self.use_moe:
                moe_pred_list.append(pair_vision_text_cross_pred)
        else:
            pair_vision_text_cross_pred = None

        if self.use_triplet_vision_text_cross:
            # 5. subj_obj_rel_triplet level, vision text cross
            position_ids = torch.zeros(
                sub_obj_pair_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_embedding.device)
            token_type_ids = torch.zeros(
                sub_obj_pair_embedding.size()[:-1], dtype=torch.long, device=sub_obj_pair_embedding.device)
            sub_obj_pair_embedding_5 = self.fc_triplet_vision_text_cross_input(
                sub_obj_pair_embedding)
            # each relation has a different text embedding
            sub_obj_rel_triplet_text_embedding_5 = self.fc_triplet_text_prompt_input(
                sub_obj_rel_triplet_text_embedding)

            batch_size, _, feature_size = sub_obj_rel_triplet_text_embedding_5.shape
            sub_obj_rel_triplet_text_embedding_5 = sub_obj_rel_triplet_text_embedding_5.reshape(
                batch_size, -1, self.num_relation_classes, feature_size)
            triplet_vision_text_cross_pred_list = []
            for rel_idx in range(self.num_relation_classes):
                vision_text_cross_output = self.sub_obj_pair_level_model(
                    inputs_embeds=sub_obj_pair_embedding_5,
                    encoder_hidden_states=sub_obj_rel_triplet_text_embedding_5[:,
                                                                               :, rel_idx, :],
                    encoder_attention_mask=sub_obj_pair_mask_attention,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                    return_dict=True,)
                vision_text_cross_embedding = vision_text_cross_output.last_hidden_state
                vision_text_cross_embedding = self.fc_triplet_vision_text_cross_output(
                    vision_text_cross_embedding)
                this_triplet_vision_text_cross_pred = self.triplet_vision_text_cross_pred[rel_idx](
                    vision_text_cross_embedding).reshape(
                    [-1, 1, object_num, object_num])
                triplet_vision_text_cross_pred_list.append(
                    this_triplet_vision_text_cross_pred)
            triplet_vision_text_cross_pred = torch.cat(
                triplet_vision_text_cross_pred_list, dim=1)
            if self.use_moe:
                moe_pred_list.append(triplet_vision_text_cross_pred)
        else:
            triplet_vision_text_cross_pred = None

        # 6. moe
        if self.use_moe:
            sub_obj_pair_embedding_moe = self.moe_input_pair_vision(
                sub_obj_pair_embedding)
            sub_obj_pair_text_embedding_moe = self.moe_input_pair_text(
                sub_obj_pair_text_embedding)
            sub_obj_rel_triplet_text_embedding_moe = self.moe_input_triplet_text(
                sub_obj_rel_triplet_text_embedding)
            batch_size, _, feature_size = sub_obj_rel_triplet_text_embedding_moe.shape
            sub_obj_rel_triplet_text_embedding_moe = sub_obj_rel_triplet_text_embedding_moe.reshape(
                batch_size, -1, self.num_relation_classes, feature_size)
            sub_obj_rel_triplet_text_embedding_moe = sub_obj_rel_triplet_text_embedding_moe.mean(
                dim=2)
            moe_output = self.moe(sub_obj_pair_embedding_moe +
                                  sub_obj_pair_text_embedding_moe +
                                  sub_obj_rel_triplet_text_embedding_moe)
            if self.moe_weight_type == 'v1':
                moe_weight = F.softmax(moe_output, dim=-1)
                moe_weight = moe_weight.permute([0, 2, 1]).reshape(
                    [batch_size, -1, object_embedding.shape[1], object_embedding.shape[1]])
            elif self.moe_weight_type == 'v2':
                moe_weight = F.softmax(
                    moe_output.reshape([batch_size, object_embedding.shape[1] * object_embedding.shape[1], self.num_relation_classes, -1]), dim=-1)
                moe_weight = moe_weight.permute([0, 3, 2, 1]).reshape(
                    [batch_size, -1, self.num_relation_classes, object_embedding.shape[1], object_embedding.shape[1]])
            else:
                assert False, 'Please use support moe weight type.'
            moe_pred = torch.zeros_like(
                moe_pred_list[0]).to(moe_pred_list[0].device)
            for idx, this_moe_pred in enumerate(moe_pred_list):
                if self.moe_weight_type == 'v1':
                    moe_pred += this_moe_pred * moe_weight[:, idx:idx+1, :, :]
                elif self.moe_weight_type == 'v2':
                    moe_pred += this_moe_pred * moe_weight[:, idx, :, :, :]
                else:
                    assert False, 'Please use support moe weight type.'
        else:
            moe_pred = None

        # 6. output
        output_dict = dict()
        output_dict['object_vision_only_pred'] = object_vision_only_pred
        output_dict['pair_vision_only_pred'] = pair_vision_only_pred
        output_dict['pair_text_vision_cross_pred'] = pair_text_vision_cross_pred
        output_dict['pair_vision_text_cross_pred'] = pair_vision_text_cross_pred
        output_dict['triplet_vision_text_cross_pred'] = triplet_vision_text_cross_pred
        output_dict['moe_pred'] = moe_pred

        return output_dict

    def loss(self, pred, target, mask_attention, prefix):
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
        losses['{}_rel_loss'.format(prefix)] = loss * self.loss_weight

        if self.loss_type != 'v0_softmax':
            # f1, p, r
            # [f1, precise, recall], [f1_mean, precise_mean,
            #                         recall_mean] = self.get_f1_p_r(pred, target, mask)
            # losses['f1'] = f1
            # losses['precise'] = precise
            # losses['recall'] = recall
            # losses['f1_mean'] = f1_mean
            # losses['precise_mean'] = precise_mean
            # losses['recall_mean'] = recall_mean

            # recall
            recall_20 = self.get_recall_N(pred, target, mask, object_num=20)
            # recall_50 = self.get_recall_N(pred, target, mask, object_num=50)
            # recall_100 = self.get_recall_N(pred, target, mask, object_num=100)
            losses['{}_recall@20'.format(prefix)] = recall_20
            # losses['recall@50'] = recall_50
            # losses['recall@100'] = recall_100

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
