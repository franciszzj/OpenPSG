from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import (
    CLIPAttention, CLIPMLP, CLIPEncoder, BaseModelOutputWithPooling, _expand_mask,
    CLIPTextTransformer, CLIPVisionTransformer,
    CLIPTextModel, CLIPVisionModel, CLIPModel)


class Adapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc_down = nn.Linear(embed_dim, embed_dim // 4)
        self.act = ACT2FN['gelu']
        self.fc_up = nn.Linear(embed_dim // 4, embed_dim)

    def forward(self, x):
        x = self.fc_down(x)
        x = self.act(x)
        x = self.fc_up(x)
        return x


class CLIPEncoderLayerWithAdapter(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps)
        # Adapter
        self.after_atten_adapter = Adapter(self.embed_dim)
        self.joint_adapter = Adapter(self.embed_dim)
        # Freeze original clip parameters
        for p in self.self_attn.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # add adapter
        hidden_states = self.after_atten_adapter(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # add adapter
        joint_hidden_states = self.joint_adapter(hidden_states)
        hidden_states = self.mlp(hidden_states) + joint_hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoderWithAdapter(CLIPEncoder):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([CLIPEncoderLayerWithAdapter(
            config) for _ in range(config.num_hidden_layers)])


class CLIPTextTransformerWithAdapter(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.encoder = CLIPEncoderWithAdapter(config)
        # Freeze
        for p in self.embeddings.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        learnable_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        if learnable_embeds is not None:
            cls_embeds = hidden_states[:, 0:1, :]
            input_embeds = hidden_states[:, 1:, :]
            hidden_states = torch.cat(
                [cls_embeds, learnable_embeds, input_embeds], dim=1)
            learnable_attention_mask = attention_mask.new_ones(
                (learnable_embeds.shape[0], learnable_embeds.shape[1]))

        input_shape = hidden_states.size()
        bsz, seq_len, embed_dim = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            if learnable_embeds is not None:
                cls_attention_mask = attention_mask[:, 0:1]
                text_attention_mask = attention_mask[:, 1:]
                attention_mask = torch.cat(
                    [cls_attention_mask, learnable_attention_mask, text_attention_mask], dim=1)
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(
                last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int,
                         device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionTransformerWithAdapter(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.encoder = CLIPEncoderWithAdapter(config)
        # Freeze
        for p in self.embeddings.parameters():
            p.requires_grad = False


class CLIPTextModelWithAdapter(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformerWithAdapter(config)
        # Initialize weights and apply final processing
        self.post_init()


class CLIPVisionModelWithAdapter(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformerWithAdapter(config)
        # Initialize weights and apply final processing
        self.post_init()


class CLIPModelWithAdapter(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformerWithAdapter(text_config)
        self.vision_model = CLIPVisionTransformerWithAdapter(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(
            []) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()
