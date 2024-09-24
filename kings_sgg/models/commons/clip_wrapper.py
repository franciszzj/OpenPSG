import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer
from kings_sgg.models.commons.clip_with_adapter import (CLIPModelWithAdapter)
from kings_sgg.models.commons.clip_with_learnable_embeds import (
    CLIPModelWithLearnableEmbeds)


class CLIPWrapper(nn.Module):
    '''
    Vision-Language Transformer Wrapper
    For CLIP-like model.
    '''

    def __init__(self,
                 pretrained_model_name_or_path,
                 use_adapter=False,
                 use_learnable_prompts=False,
                 model_keep='both'):
        super().__init__()
        if use_adapter:
            self.model = CLIPModelWithAdapter.from_pretrained(
                pretrained_model_name_or_path)
        elif use_learnable_prompts:
            self.model = CLIPModelWithLearnableEmbeds.from_pretrained(
                pretrained_model_name_or_path)
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        if model_keep == 'vision':
            self.embed_dim = self.model.vision_embed_dim
            del self.model.text_model
            del self.model.text_projection
        elif model_keep == 'text':
            self.embed_dim = self.model.text_embed_dim
            del self.model.vision_model
            del self.model.visual_projection

    def forward_embeds_vision(self, embeds, attention_mask=None):
        embeds = self.model.vision_model.pre_layrnorm(embeds)
        v_outptus = self.model.vision_model.encoder(
            embeds, attention_mask=attention_mask, return_dict=True)['last_hidden_state']
        return v_outptus

    def forward_embeds_text(self, embeds, attention_mask=None):
        t_outputs = self.model.text_model.encoder(embeds, attention_mask)
        t_outputs = self.model.text_model.final_layer_norm(t_outputs[0])
        return t_outputs

    def forward_texts(self, texts, learnable_embeds=None):
        # Tokenize
        inputs = self.tokenizer(
            texts, padding=True, return_tensors='pt')
        if 'token_type_ids' in inputs.keys():
            inputs.pop('token_type_ids')
        max_position_embeddings = self.model.text_model.config.max_position_embeddings
        inputs['input_ids'] = inputs['input_ids'][:, :max_position_embeddings]
        inputs['attention_mask'] = inputs['attention_mask'][:,
                                                            :max_position_embeddings]
        # Learnable Prompts
        if learnable_embeds is not None:
            inputs['learnable_embeds'] = learnable_embeds
        for key, value in inputs.items():
            inputs[key] = value.to(self.model.device)
        # Tranformer Forward
        outputs = self.model.text_model(**inputs)
        output_embed = outputs['pooler_output']
        return output_embed
