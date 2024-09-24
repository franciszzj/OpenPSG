from pathlib import Path
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet.models.builder import HEADS
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from kings_sgg.models.commons.llama import LlamaTransformer, ModelArgs
from kings_sgg.models.commons.llama_tokenizer import LlamaTokenizer


@HEADS.register_module()
class RelationTransformerHeadV3(BaseModule):
    def __init__(self,
                 # used for llm
                 llm_path=None,
                 tokenizer_path=None,
                 max_seq_len=512,
                 special_tokens=['<|object|>', '<|relation|>'],
                 shave_language_decoder_at=6,
                 # If use fp16, the loss will be nan.
                 fpdtype=torch.FloatTensor,
                 is_inference=False,
                 causal_mask=True,
                 # not use
                 use_object_vision_only=True,
                 use_pair_vision_only=False,
                 use_pair_text_vision_cross=False,
                 use_pair_vision_text_cross=False,
                 use_triplet_vision_text_cross=False,
                 use_moe=False,
                 # used for relation head
                 ov_relation=False,
                 relation_classes=[],
                 sub_obj_merge_type='concat',
                 num_object_in_layers=0,
                 num_object_out_layers=0,
                 num_relation_out_layers=0,
                 input_feature_size=256,
                 former_feature_size=768,
                 llm_feature_size=4096,
                 output_feature_size=512,
                 num_object_classes=133,
                 # num_relation_classes=56,
                 max_object_num=30,
                 embedding_add_cls=True,
                 merge_cls_type='add',
                 positional_encoding=None,
                 use_background_feature=False,
                 loss_type='v1',
                 loss_weight=50,
                 loss_alpha=1,
                 **kwargs):
        super().__init__()
        ##########
        # Config #
        ##########
        # used for llm
        self.causal_mask = causal_mask

        # not use
        self.use_object_vision_only = use_object_vision_only
        self.use_pair_vision_only = use_pair_vision_only
        self.use_pair_text_vision_cross = use_pair_text_vision_cross
        self.use_pair_vision_text_cross = use_pair_vision_text_cross
        self.use_triplet_vision_text_cross = use_triplet_vision_text_cross
        self.use_moe = use_moe

        # used for relation head
        self.ov_relation = ov_relation
        self.relation_classes = relation_classes
        self.sub_obj_merge_type = sub_obj_merge_type  # concat, multiply
        self.input_feature_size = input_feature_size
        self.former_feature_size = former_feature_size
        self.llm_feature_size = llm_feature_size
        self.output_feature_size = output_feature_size
        self.num_object_classes = num_object_classes
        self.num_relation_classes = len(relation_classes)
        self.max_object_num = max_object_num
        self.embedding_add_cls = embedding_add_cls
        self.merge_cls_type = merge_cls_type
        self.positional_encoding = positional_encoding
        self.use_background_feature = use_background_feature
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.loss_alpha = loss_alpha

        relation_str = ''
        for relation in self.relation_classes:
            relation_str += '<|relation|> {} '.format(relation)
        self.relation_str = relation_str.strip()

        #########
        # Model #
        #########
        if not ov_relation:
            self.llama_model, self.llama_tokenizer, self.llama_args = load_llama(
                llm_path, tokenizer_path, max_seq_len, special_tokens, shave_language_decoder_at, fpdtype, is_inference)
            self.fc_object_vision_only_input = nn.Sequential(
                nn.Linear(self.input_feature_size, self.llm_feature_size),
                nn.LayerNorm(llm_feature_size),)
            self.fc_object_vision_only_output = nn.Sequential(
                nn.Linear(llm_feature_size, output_feature_size),
                nn.LayerNorm(output_feature_size),)
            self.object_vision_only_sub_pred = nn.Linear(output_feature_size,
                                                         output_feature_size * self.num_relation_classes)
            self.object_vision_only_obj_pred = nn.Linear(output_feature_size,
                                                         output_feature_size * self.num_relation_classes)
        else:
            self.o_in_former = BertEncoder(BertConfig(
                num_hidden_layers=num_object_in_layers))
            self.o_out_former = BertEncoder(BertConfig(
                num_hidden_layers=num_object_out_layers))
            self.r_out_former = BertEncoder(BertConfig(
                num_hidden_layers=num_relation_out_layers))

            self.llama_model, self.llama_tokenizer, self.llama_args = load_llama(
                llm_path, tokenizer_path, max_seq_len, special_tokens, shave_language_decoder_at, fpdtype, is_inference)
            self.fc_object_in_former = nn.Sequential(
                nn.Linear(self.input_feature_size, self.former_feature_size),
                nn.LayerNorm(self.former_feature_size),)
            self.fc_object_former_to_llm = nn.Sequential(
                nn.Linear(self.former_feature_size, self.llm_feature_size),
                nn.LayerNorm(self.llm_feature_size),)
            self.fc_object_llm_to_former = nn.Sequential(
                nn.Linear(self.llm_feature_size, self.former_feature_size),
                nn.LayerNorm(self.former_feature_size),)
            self.fc_object_out_former = nn.Sequential(
                nn.Linear(self.former_feature_size, self.output_feature_size),
                nn.LayerNorm(self.output_feature_size),)
            self.fc_relation_llm_to_former = nn.Sequential(
                nn.Linear(self.llm_feature_size, self.former_feature_size),
                nn.LayerNorm(self.former_feature_size),)
            if self.sub_obj_merge_type == 'concat':
                self.relation_output_size = self.output_feature_size * 2 * 1 + 1
            elif self.sub_obj_merge_type == 'multiply':
                self.relation_output_size = self.output_feature_size * 1 + 1
            else:
                assert False, 'Please use support sub_obj_merge_type: concat, multiply, but you are using: {}'.format(
                    self.sub_obj_merge_type)
            self.fc_relation_out_former = nn.Sequential(
                nn.Linear(self.former_feature_size, self.relation_output_size),
                nn.LayerNorm(self.relation_output_size),)
            self.fc_sub_pred = nn.Linear(
                output_feature_size, output_feature_size)
            self.fc_obj_pred = nn.Linear(
                output_feature_size, output_feature_size)

        ########
        # Loss #
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
        sub_obj_pair_name_list = inputs['sub_obj_pair_name_list']
        sub_obj_pair_str_list = inputs['sub_obj_pair_str_list']
        sub_obj_pair_text_embedding = inputs['sub_obj_pair_text_embedding']
        sub_obj_pair_mask_attention = inputs['sub_obj_pair_mask_attention']
        sub_obj_rel_triplet_text_embedding = inputs['sub_obj_rel_triplet_text_embedding']

        if not self.ov_relation:
            object_embedding = self.fc_object_vision_only_input(
                object_embedding)
            object_embedding = self.llama_model(
                object_embedding, is_train=self.training, causal_mask=self.causal_mask)[1]
            object_embedding = self.fc_object_vision_only_output(
                object_embedding)
            sub_embedding = self.object_vision_only_sub_pred(object_embedding).reshape(
                [batch_size, object_num, self.num_relation_classes, self.output_feature_size]).permute([0, 2, 1, 3])
            obj_embedding = self.object_vision_only_obj_pred(object_embedding).reshape(
                [batch_size, object_num, self.num_relation_classes, self.output_feature_size]).permute([0, 2, 1, 3])
            relation_pred = torch.einsum(
                'nrsc,nroc->nrso', sub_embedding, obj_embedding)
        else:
            # build object embedding
            object_embedding = self.fc_object_in_former(object_embedding)
            object_embedding = self.o_in_former(
                object_embedding).last_hidden_state
            object_embedding = self.fc_object_former_to_llm(object_embedding)
            object_size = object_embedding.shape[1]
            # build relation embedding
            relation_ids = self.llama_tokenizer.encode(
                self.relation_str, bos=False, eos=False)
            rel_sep_id = self.llama_tokenizer.special_tokens_ttoi['<|relation|>']
            relation_begin_idxes = np.where(relation_ids == rel_sep_id)[0]
            relation_ids = torch.tensor(relation_ids).unsqueeze(0).to(
                object_embedding.device)
            relation_embedding = self.llama_model.tok_embeddings(relation_ids)
            relation_size = relation_embedding.shape[1]
            # construct input embedding
            input_embedding = torch.cat(
                [object_embedding, relation_embedding], dim=1)

            # llm forward
            output_embedding = self.llama_model(
                input_embedding, is_train=self.training, causal_mask=self.causal_mask)[1]
            object_embedding = output_embedding[:, :object_size]
            relation_embedding = output_embedding[:, object_size:]

            # process object embedding
            object_embedding = self.fc_object_llm_to_former(object_embedding)
            object_embedding = self.o_out_former(
                object_embedding).last_hidden_state
            object_embedding = self.fc_object_out_former(object_embedding)
            # process relation embedding
            # merge same relation embedding
            relation_merged_embedding = torch.zeros(
                [batch_size, self.num_relation_classes, self.llm_feature_size]).to(relation_embedding.device)
            for idx, rel_idx in enumerate(relation_begin_idxes):
                if idx == len(relation_begin_idxes) - 1:
                    relation_merged_embedding[:, idx] = \
                        object_embedding[:, rel_idx:].mean(dim=1)
                else:
                    relation_merged_embedding[:, idx] = \
                        object_embedding[:,
                                         rel_idx:relation_begin_idxes[idx+1]].mean(dim=1)
            relation_merged_embedding = self.fc_relation_llm_to_former(
                relation_merged_embedding)
            relation_merged_embedding = self.r_out_former(
                relation_merged_embedding).last_hidden_state
            relation_merged_embedding = self.fc_relation_out_former(
                relation_merged_embedding)[0]
            relation_weight, relation_bias = torch.split(
                relation_merged_embedding, [self.relation_output_size - 1, 1], dim=1)
            relation_bias = relation_bias.squeeze(1)

            sub_embedding = self.fc_sub_pred(object_embedding)
            obj_embedding = self.fc_obj_pred(object_embedding)

            if self.sub_obj_merge_type == 'concat':
                sub_embedding = torch.repeat_interleave(
                    sub_embedding, object_num, dim=1)
                obj_embedding = obj_embedding.repeat([1, object_num, 1])
                sub_obj_embedding = torch.cat(
                    [sub_embedding, obj_embedding], dim=2)
                relation_pred = F.linear(
                    sub_obj_embedding, relation_weight, relation_bias)
                relation_pred = relation_pred.reshape(
                    [batch_size, object_num, object_num, self.num_relation_classes])
            elif self.sub_obj_merge_type == 'multiply':
                sub_obj_embedding = torch.einsum(
                    'nsc,noc->nsoc', sub_embedding, obj_embedding)
                relation_pred = F.linear(
                    sub_obj_embedding, relation_weight, relation_bias)
            relation_pred = relation_pred.permute([0, 3, 1, 2])

        # 6. output
        output_dict = dict()
        output_dict['object_vision_only_pred'] = relation_pred
        output_dict['pair_vision_only_pred'] = None
        output_dict['pair_text_vision_cross_pred'] = None
        output_dict['pair_vision_text_cross_pred'] = None
        output_dict['triplet_vision_text_cross_pred'] = None
        output_dict['moe_pred'] = None

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


def load_llama(llama_path=None, tokenizer_path=None, max_seq_len=512, special_tokens=("<|image|>",), shave_language_decoder_at=6, fpdtype=torch.HalfTensor, is_inference=True, device=None):

    with open(Path(llama_path) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, **params)
    # args.embed_dim = model_args.dim
    print(f"LLaMA model params: {vars(model_args)}")

    tokenizer = LlamaTokenizer(model_path=tokenizer_path)

    # pad a special token <|empty|> to the end of the original embedding matrix
    # for the negative index of tokenizer.pad_id, i.e., -1
    special_tokens = list(special_tokens) + ["<|empty|>"]
    if len(special_tokens) > 0:
        tokenizer.add_special_tokens(special_tokens)
    model_args.vocab_size = tokenizer.n_words
    model_args.n_special_tokens = tokenizer.n_special_tokens
    model_args.shave_language_decoder_at = shave_language_decoder_at

    # load with fp16 or bf16
    torch.set_default_tensor_type(fpdtype)

    model = LlamaTransformer(model_args)

    # switch back to fp32
    torch.set_default_tensor_type(torch.FloatTensor)

    if not is_inference:
        checkpoints = sorted(Path(llama_path).glob("*.pth"))

        # TODO: support to load multiple checkpoints for LLaMA-13B / 70B
        if len(checkpoints) == 1:
            ckpt_path = checkpoints[0]
        else:
            raise ValueError(
                f"currently only support one checkpoint, got {len(checkpoints)}"
            )

        # print(
        #     f"loading pre-trained checkpoints of LLaMA {ckpt_path} on device {device}"
        # )
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if len(special_tokens) > 0:
            # pad the special tokens to the end of the original embedding matrix
            k = "tok_embeddings.weight"
            n_dim = checkpoint[k].shape[-1]
            v = torch.empty(tokenizer.n_words, n_dim).normal_(mean=0.0, std=1)
            v[: -tokenizer.n_special_tokens].copy_(checkpoint[k])
            checkpoint[k] = v

            k = "output.weight"
            n_dim = checkpoint[k].shape[-1]
            v = torch.zeros(tokenizer.n_words, n_dim)
            if tokenizer.n_special_tokens > 1:
                # largely have special tokens other than <|empty|>
                nn.init.kaiming_uniform_(v[:-1], a=math.sqrt(5))
            v[: -tokenizer.n_special_tokens].copy_(checkpoint[k])
            checkpoint[k] = v

            del v

        msgs = model.load_state_dict(checkpoint, strict=False)
        if len(msgs.missing_keys) > 0:
            print("-", msgs.missing_keys)

        del checkpoint
    else:
        print("inference mode is on, skip loading pre-trained checkpoints of LLaMA")

    # model = model.to(device)

    return model, tokenizer, model_args
