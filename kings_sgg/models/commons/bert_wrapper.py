import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer
from kings_sgg.models.commons.bert_with_adapter import (
    BertModelWithAdapter)
from kings_sgg.models.commons.bert_with_learnable_embeds import (
    BertModelWithLearnableEmbeds)


class BertWrapper(nn.Module):
    '''
    Language Transformer Wrapper
    For Bert-like model.
    '''

    def __init__(self,
                 pretrained_model_name_or_path,
                 load_pretrained_weights=True,
                 num_transformer_layer='all',
                 use_adapter=False,
                 use_learnable_prompts=False,
                 add_cross_attention=False,):
        super().__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path)
        if add_cross_attention:
            config.is_decoder = True
            config.add_cross_attention = True
        if use_adapter:
            self.model = BertModelWithAdapter.from_pretrained(
                pretrained_model_name_or_path, config=config)
        elif use_learnable_prompts:
            self.model = BertModelWithLearnableEmbeds.from_pretrained(
                pretrained_model_name_or_path, config=config)
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            if load_pretrained_weights:
                self.model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path, config=config)
            else:
                self.model = AutoModel.from_config(config)
        if num_transformer_layer != 'all' and isinstance(num_transformer_layer, int):
            self.model.encoder.layer = self.model.encoder.layer[:num_transformer_layer]
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def forward_embeds(self, embeds, attention_mask=None, position_ids=None):
        outputs = self.model(inputs_embeds=embeds,
                             attention_mask=attention_mask,
                             position_ids=position_ids)
        output_embeds = outputs['last_hidden_state']
        return output_embeds

    def forward_texts(self, texts, learnable_embeds=None):
        inputs = self.tokenizer(
            texts, padding=True, return_tensors='pt')
        if learnable_embeds is not None:
            inputs['learnable_embeds'] = learnable_embeds
        for key, value in inputs.items():
            inputs[key] = value.to(self.model.device)
        outputs = self.model(**inputs)
        output_embed = outputs['pooler_output']
        return output_embed
