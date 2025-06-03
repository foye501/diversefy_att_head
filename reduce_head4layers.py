import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention

class ReducedHeadAttention(GPT2Attention):
    def __init__(self, config, n_head_custom):
        super().__init__(config)
        # Override number of heads and embed dim
        self.num_heads = n_head_custom
        self.split_size = self.num_heads * (config.n_embd // config.n_head)

        # Adjust projections
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.split_size)
        self.c_proj = nn.Linear(self.split_size, self.embed_dim)

class CustomBlock(GPT2Block):
    def __init__(self, config, n_head_custom):
        super().__init__(config)
        self.reduced_attn_in_proj = nn.Linear(config.n_embd, n_head_custom * (config.n_embd // config.n_head))
        self.attn = ReducedHeadAttention(config, n_head_custom)
        self.reduced_attn_out_proj = nn.Linear(n_head_custom * (config.n_embd // config.n_head), config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, **kwargs):
        residual = hidden_states

        # Pre-attention projection
        reduced_input = self.reduced_attn_in_proj(hidden_states)
        attn_outputs = self.attn(reduced_input, layer_past=layer_past, attention_mask=attention_mask)
        attn_output = attn_outputs[0]
        attn_output = self.reduced_attn_out_proj(attn_output)

        hidden_states = residual + attn_output
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + feed_forward_hidden_states
        return (hidden_states,) + attn_outputs[1:]

class CustomGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

        layer_custom_heads = {
            1: 3,
            7: 3,
            8: 5,
            9: 3,
            10: 5,
            11: 3
        }

        for i, n_head in layer_custom_heads.items():
            self.transformer.h[i] = CustomBlock(config, n_head_custom=n_head)

# Example usage
if __name__ == "__main__":
    config = GPT2Config()
    model = CustomGPT2(config)

    print(model)  # Print the structure to verify changes
