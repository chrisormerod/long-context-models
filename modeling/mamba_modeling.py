# modeling_mamba.py
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class MambaForSequenceClassification(PreTrainedModel):
    """
    MambaForSequenceClassification

    A lightweight sequence-classification wrapper for the Mamba base model hosted at:
      https://huggingface.co/state-spaces/mamba-130m-hf

    Behavior:
      - Delegates base-model loading to AutoModel so HF's model registry handles the correct
        underlying implementation.
      - Applies a dropout + linear classification head to either:
          * the base model's `pooler_output` (if available), or
          * the mean-pooled token embeddings (masked by attention_mask).
      - Computes CrossEntropyLoss when `labels` is provided.

    Notes:
      - `config` should be an AutoConfig instance whose `hidden_size` attribute is present.
      - To load pre-trained weights, use the inherited `from_pretrained` method:
            MambaForSequenceClassification.from_pretrained("state-spaces/mamba-130m-hf", num_labels=2)
    """

    config_class = AutoConfig  # informative; actual config will be an AutoConfig-derived object
    base_model_prefix = "mamba"

    def __init__(self, config):
        """
        Args:
            config: a transformers config object. The `num_labels` attribute will be read from config
                    if present; otherwise you can pass `num_labels` to from_pretrained via kwargs.
        """
        super().__init__(config)

        # number of labels for classification (allow config to contain it)
        num_labels = getattr(config, "num_labels", None)
        if num_labels is None:
            # default to binary if not provided (user may override via kwargs in `from_pretrained`)
            num_labels = 2
            config.num_labels = num_labels

        self.num_labels = num_labels

        # Base model: use AutoModel so HF resolves the correct Mamba implementation.
        # When constructing with `from_pretrained`, the library will call this __init__ with a config,
        # then PreTrainedModel.from_pretrained will replace the model weights appropriately.
        self.mamba = AutoModel.from_config(config)

        # Dropout: prefer config.classifier_dropout or config.hidden_dropout_prob if available
        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # Classifier head
        hidden_size = getattr(config, "hidden_size")
        if hidden_size is None:
            # some configs may use d_model or hidden_dim; try common fallbacks
            hidden_size = getattr(config, "d_model", None) or getattr(config, "hidden_dim", None)
            if hidden_size is None:
                raise ValueError("Config must define hidden_size (or d_model / hidden_dim).")

        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()  # calls _init_weights and other HF utilities

    def _mean_pooling(self, token_embeddings: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        """
        Mean pooling across token embeddings, taking attention mask into account.

        Args:
            token_embeddings: (batch, seq_len, hidden)
            attention_mask: (batch, seq_len) or None

        Returns:
            pooled (batch, hidden)
        """
        if attention_mask is None:
            return token_embeddings.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # (batch, seq_len, 1)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[Tensor, Tensor], SequenceClassifierOutput]:
        """
        Forward pass.

        Accepts the usual transformer model inputs and returns logits (and loss if labels provided).
        Uses pooler_output if the base model provides it, otherwise mean-pools last_hidden_state.

        Returns:
            SequenceClassifierOutput when return_dict=True (default), otherwise tuple (loss, logits, ...)
        """
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        # Pass through to the base Mamba model.
        # We use `return_dict=True` for easier handling of outputs in modern HF style.
        base_outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        # Some models provide `pooler_output`; others do not.
        pooled_output = None
        if getattr(base_outputs, "pooler_output", None) is not None:
            pooled_output = base_outputs.pooler_output
        else:
            # Fall back to mean pooling of last_hidden_state
            last_hidden_state = base_outputs.last_hidden_state  # (batch, seq_len, hidden)
            pooled_output = self._mean_pooling(last_hidden_state, attention_mask)  # (batch, hidden)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # classification loss (supports multi-class integer labels)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + tuple(v for v in base_outputs.to_tuple()[1:])  # keep compatibility
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(base_outputs, "hidden_states", None),
            attentions=getattr(base_outputs, "attentions", None),
        )

    # Optional: convenience from_pretrained helper to allow num_labels override easily
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """
        Override only to allow passing num_labels in kwargs conveniently,
        but otherwise use the super implementation which will:
          - instantiate config via AutoConfig.from_pretrained(...)
          - call cls(config, **init_kwargs)
          - load weights into the model
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)