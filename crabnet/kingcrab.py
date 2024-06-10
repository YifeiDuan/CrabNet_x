import numpy as np
import pandas as pd

import torch
from torch import nn

import copy
from typing import Optional, Any, Union, Callable

import warnings
from torch.nn import functional as F
from torch import Tensor
# from torch.nn.activation import MultiheadAttention
# from torch.nn.dropout import Dropout
# from torch.nn.linear import Linear
# from torch.nn.normalization import LayerNorm

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32 #test
main_dir = "/home/jupyter/YD/MTENCODER/CrabNet__/"


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

class new_TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(new_TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(new_TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)





class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims, branched_ffnn, dropout):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim] + hidden_layer_dims

        self.branched_ffnn = branched_ffnn

        if dropout:
            print('Dropout is used!')
            self.dropout = nn.ModuleList([nn.Dropout(dropout)
                                   for i in range(len(dims) - 1)])
        else:
            self.dropout = None

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                  for i in range(len(dims) - 1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                      if (dims[i] != dims[i + 1])
                                      else nn.Identity()
                                      for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        if self.branched_ffnn:
            for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
                fea = act(fc(fea))+res_fc(fea)
        else:
            if self.dropout:
                for fc, act, drop in zip(self.fcs, self.acts, self.dropout):
                    fea = act(fc(drop(fea)))
            else:
                for fc, act in zip(self.fcs, self.acts):
                    fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Embedder(nn.Module):
    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = compute_device

        elem_dir = main_dir + 'data/element_properties'
        # # Choose what element information the model receives
        mat2vec = f'{elem_dir}/mat2vec.csv'  # element embedding
        # mat2vec = f'{elem_dir}/onehot.csv'  # onehot encoding (atomic number)
        # mat2vec = f'{elem_dir}/random_200.csv'  # random vec for elements

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))

        cpd = torch.randn((3, feat_size))
        cpd = cpd / np.sqrt(feat_size)
        cat_array = np.concatenate([zeros, cbfv, cpd])
        cat_array = torch.as_tensor(cat_array, dtype=data_type_torch)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=data_type_torch)
        self.cfv = nn.Embedding(cbfv.shape[0] + 1, self.d_model)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(
            50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(
            50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            # clamp x[x > 1] = 1
            x = torch.clamp(x, max=1)
            # x = 1 - x  # for sinusoidal encoding at x=0
        # clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1 / self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out

class FractionalEncoderNumeral(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.compute_device = compute_device
        self.project = nn.Linear(1, d_model).to(self.compute_device)
        self.act = nn.LeakyReLU()


    def forward(self, x):
        x = x.clone()
        x = self.act(self.project(x.unsqueeze(dim=-1)))
        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 encoder_ff,
                 numeral_embeddings,
                 special_tok_zero_fracs=False,
                 frac=False,
                 attn=True,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.heads = heads
        self.fractional = frac
        self.attention = attn
        self.special_tok_zero_fracs = special_tok_zero_fracs
        self.numeral_embeddings = numeral_embeddings
        if self.numeral_embeddings: 
            print('\n Learned numeral_embeddings!')
        self.compute_device = compute_device
        self.embed = Embedder(d_model=self.d_model,
                              compute_device=self.compute_device)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.numeral = FractionalEncoderNumeral(self.d_model)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))

        if self.attention:
            encoder_layer = new_TransformerEncoderLayer(self.d_model,
                                                       nhead=self.heads,
                                                       dim_feedforward=encoder_ff,
                                                       dropout=0.1)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=self.N)

    def forward(self, src, frac):
        x = self.embed(src) * 2 ** self.emb_scaler

        attention = src.clone()
        attention[attention != 0] = 1

        mask = attention
        src_mask = ~attention.bool()

        """mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1"""

        if self.numeral_embeddings:
            frac_embed = self.numeral(frac)

        else:
            pe = torch.zeros_like(x)
            ple = torch.zeros_like(x)
            pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
            ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
            pe[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
            ple[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler
            frac_embed = pe + ple

        if self.special_tok_zero_fracs == 'both':
            frac_embed[:, 0, :] = 0
            frac_embed[:, -1, :] = 0
        
        if self.special_tok_zero_fracs == 'cpd':
            frac_embed[:, 0, :] = 0
        
        if self.special_tok_zero_fracs == 'eos':
            frac_embed[:, -1, :] = 0

        if self.attention:
            x_src = x + frac_embed
            x_src = x_src.transpose(0, 1)
            x = self.transformer_encoder(x_src,
                                         src_key_padding_mask=src_mask)
            x = x.transpose(0, 1)

        if self.fractional:
            x = x * frac.unsqueeze(2).repeat(1, 1, self.d_model)

        hmask = attention.unsqueeze(-1).expand(-1, -1, self.d_model)
        #hmask = mask[:, :, 0:1].repeat(1, 1, self.d_model)
        if mask is not None:
            x = x.masked_fill(hmask == 0, 0)

        return x


class CrabNet(nn.Module):
    def __init__(self,
                 config,
                 compute_device=None
                 ):
        super().__init__()
        super(CrabNet, self).__init__()

        self.config = config['model']

        self.decoder = self.config['decoder']

        valid_options = ["mask prediction", "multi-task", "meanpool", "cpd", "roost", "predict"]
        assert self.decoder in valid_options, f"{self.decoder} is not a valid option. Please select from {valid_options}."

        self.d_model = self.config['d_model']
        self.N = self.config['N']
        self.heads = self.config['heads']
        self.residual_nn_dim = self.config['residual_nn_dim']
        self.dropout = self.config['dropout']
        self.branched_ffnn = self.config['branched_ffnn']
        self.encoder_ff = self.config['encoder_ff']
        self.special_tok_zero_fracs = self.config['special_tok_zero_fracs']
        self.numeral_embeddings = self.config['numeral_embeddings']
        self.task_list = config['task_list']
        self.task_types = config['task_types']

        if self.special_tok_zero_fracs:
            print('\n special_tok_zero_fracs!')

        self.compute_device = compute_device
        self.encoder = Encoder(d_model=self.d_model,
                               N=self.N,
                               heads=self.heads,
                               encoder_ff=self.encoder_ff,
                               special_tok_zero_fracs=self.special_tok_zero_fracs,
                               compute_device=self.compute_device,
                               numeral_embeddings=self.numeral_embeddings)
        if isinstance(self.task_list, list):
            modules = []
            for task_type in self.task_types:
                if task_type=='MLM':
                    print('MLM')
                    modules.append(ResidualNetwork(self.d_model,
                                  119,
                                  self.residual_nn_dim,
                                  self.branched_ffnn,
                                  .2))
                if task_type=='REG':
                    print('REG')
                    modules.append(ResidualNetwork(self.d_model,
                                  2,
                                  self.residual_nn_dim,
                                  self.branched_ffnn,
                                  self.dropout))
                if task_type=='BIN':
                    print('BIN')
                    modules.append(ResidualNetwork(self.d_model,
                                  1,
                                  self.residual_nn_dim,
                                  self.branched_ffnn,
                                  self.dropout))    
                if task_type=='MCC':
                    print(task_type)
                    modules.append(ResidualNetwork(self.d_model,
                                  300,
                                  self.residual_nn_dim,
                                  self.branched_ffnn,
                                  self.dropout))
                if task_type=='MREG':
                    print(task_type)
                    modules.append(ResidualNetwork(self.d_model,
                                  18,
                                  self.residual_nn_dim,
                                  self.branched_ffnn,
                                  self.dropout))
            self.output_nn = nn.ModuleList(modules)

        else:
            self.output_nn = ResidualNetwork(self.d_model,
                                         2,
                                         self.residual_nn_dim,
                                         self.branched_ffnn,
                                         self.dropout)
            
    def get_number_classes(self):
        return

    def forward(self, input, embeddings=False):
        # 1 Get an embedding for the compound
        output = self.encoder(input['src_masked'], input['frac'])
        # mask so you only average "elements"
        mask = (input['src_masked'] == 0).unsqueeze(-1).repeat(1, 1, self.d_model)
        output = output.masked_fill(mask, 0)

        if embeddings:
            return output[:, 0, :]

        res = {}
        # 2 decode
        if self.decoder == 'roost':
            output = self.output_nn(output)
            mask = (input['src'] == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
            output = output.masked_fill(mask, 0)

            output = output.sum(dim=1)/(~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output_pred = output * probability

        if self.decoder == 'meanpool':
            output = output.mean(dim=1)
            output_pred = self.output_nn(output)

        if self.decoder == 'cpd':
            output = output[:, 0, :]
            output_pred = self.output_nn(output)
            return output_pred

        #todo if masking is used
        # 1. get an encoding for the compound
        # 2. identify the number of masked tokens
        # 3. push each mask encoding through the decoder layers

        if self.decoder == 'mask prediction':
            # get all msk positions
            indices = torch.where(input['position_mask'] == 1)

            # save clsf output and targets to dict
            res['targets'] = input['src'][indices]

            masked_embeddings = output[indices]
            output_pred = self.output_nn(masked_embeddings)
            return output_pred


        if self.decoder == 'multi-task':
            for task_id in range(len(self.output_nn)):
                res[task_id] = {}

            for task_id in range(len(self.output_nn)):
                mask = torch.where(input['tasks'] == task_id)[0]

                if self.task_types[task_id] == 'MLM':
                    indices = input['position_mask'][mask] == 1
                    task_output = self.output_nn[task_id](output[mask][indices])
                    res[task_id]['y_true'] = input['src'][mask][indices]
                else:
                    task_output = self.output_nn[task_id](output[mask, 0, :])

                res[task_id]['y_pred'] = task_output
            return res

        if self.decoder == 'predict':
            print('decoder')
            output_pred = torch.zeros(output.shape[0], len(self.output_nn), 2).to(input['src_masked'].device)

            for task_id in range(len(self.output_nn)):
                # Forward pass for the specific task network
                task_output = self.output_nn[task_id](output[:, 0, :])

                # Check the output shape
                if task_output.shape != (output.shape[0], 2):
                    continue  # Skip this iteration and move to the next one
                else:
                    # Store the output in the corresponding slice of output_pred
                    output_pred[:, task_id] = task_output
            return output_pred


if __name__ == '__main__':
    model = CrabNet()
