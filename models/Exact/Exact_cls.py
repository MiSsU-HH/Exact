import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.Exact.module import Attention, PreNorm, FeedForward, TemporalAwareAffinityPropagationModule
from lib.modules import momentum_update, distributed_sinkhorn, l2_normalize
from timm.models.layers import trunc_normal_

class Transformer(nn.Module):
    """
    Transformer block consisting of multi-head self-attention and MLP layers.
    Return self-attention matrix while return_att is True.
    """ 
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., return_att=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.return_att = return_att
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, return_att=return_att)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        if self.return_att:
            attn_weights = []
            for attn, ff in self.layers:
                x, weights_i = attn(x)
                attn_weights.append(weights_i)
                x = ff(x) + x
            attn_weights = torch.stack(attn_weights)
            attn_weights = torch.mean(attn_weights, dim=2)
            return self.norm(x), attn_weights
        else:
            for attn, ff in self.layers:
                x = attn(x)
                x = ff(x) + x
            return self.norm(x)

class Exact_cls(nn.Module):
    """
    Temporal-Spatial Vision Transformer for Time series remote sensing image classification.
    Includes prototype learning and Temporal-Aware Affinity Propagation.
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.dim = model_config['dim']
        self.temporal_depth = model_config['temporal_depth']
        self.spatial_depth = model_config['spatial_depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),
        )
        self.to_temporal_embedding_input = nn.Linear(365, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(
            self.dim, self.temporal_depth, self.heads, self.dim_head,
            self.dim * self.scale_dim, self.dropout, return_att=True
        )
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(
            self.dim, self.spatial_depth, self.heads, self.dim_head,
            self.dim * self.scale_dim, self.dropout, return_att=False
        )
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.patch_head = nn.Conv2d(self.dim, 1, kernel_size=3, stride=1, padding=1)
        self.taap_module = TemporalAwareAffinityPropagationModule(num_iter=3, dilations=[2])
        self.num_prototype = 2
        self.fg_proto = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, self.dim), requires_grad=True)
        self.bg_proto = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, self.dim), requires_grad=True)
        trunc_normal_(self.fg_proto, std=0.02)
        trunc_normal_(self.bg_proto, std=0.02)

    def prototype_learning(self, fea, cams, cls_label_gt, is_bg=False):
        """
        Update foreground or background prototypes.
        """
        protos = self.bg_proto.data.clone() if is_bg else self.fg_proto.data.clone()
        proto_entropy = torch.zeros(self.num_classes).to(fea.device)
        proto_dis = torch.zeros(self.num_classes).to(fea.device)
        for k in range(self.num_classes):
            if not is_bg:
                b_indices = torch.nonzero(cls_label_gt[:, k] == 1).squeeze(-1)
                if len(b_indices) == 0:
                    continue           
                cam_k = cams[b_indices,k].contiguous().view(-1)
                binary_mask = cam_k >= 0.4
                fea_k = fea[b_indices,k].clone()   
            else:
                cam_k = cams[:,k].contiguous().view(-1)
                binary_mask = cam_k < 0.2
                fea_k = fea[:,k].clone()
            fea_k = rearrange(fea_k, 'b c h w -> (b h w) c')
            fea_k = l2_normalize(fea_k)
            init_q = torch.einsum('nd,md->nm', fea_k, self.bg_proto[k] if is_bg else self.fg_proto[k])
            init_q = init_q[binary_mask, ...]
            if init_q.shape[0] == 0:
                continue
            q, indexs = distributed_sinkhorn(init_q)
            m_k = cam_k[binary_mask]
            c_k = fea_k[binary_mask, ...]
            fg_cosine_sim = F.cosine_similarity(c_k.unsqueeze(1), self.fg_proto.view(-1,self.fg_proto.shape[-1]).squeeze(0), dim=2) / 0.1
            bg_cosine_sim = F.cosine_similarity(c_k.unsqueeze(1), self.bg_proto.view(-1,self.bg_proto.shape[-1]).squeeze(0), dim=2) / 0.1      
            cosine_sim = torch.cat((fg_cosine_sim, bg_cosine_sim), dim=-1)
            if not is_bg:
                indexs += self.num_prototype * k
            else:
                indexs += (self.num_prototype * self.num_classes + self.num_prototype * k)
            proto_entropy[k] = F.cross_entropy(cosine_sim, indexs)
            logits = torch.gather(cosine_sim, 1, indexs.view(-1, 1))
            proto_dis[k] = (1 - logits).pow(2).mean()
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile  
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  
            f = m_q.transpose(0, 1) @ c_q 
            n = torch.sum(m_q, dim=0)
            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :], momentum=0.999, debug=False)
                protos[k, n != 0, :] = new_value
        if not is_bg:
            self.fg_proto = nn.Parameter(l2_normalize(protos), requires_grad=False)
        else:
            self.bg_proto = nn.Parameter(l2_normalize(protos), requires_grad=False)
        return proto_entropy.mean(), proto_dis.mean()

    def forward(self, x, cls_label_gt=None, abs_step=0, generate_cam=False):
        """
        Forward pass for classification, CAM generation, and optional prototype refinement.

        Args:
            x (Tensor): Input tensor of shape [B, T, H, W, C].
            cls_label_gt (Tensor, optional): One-hot ground truth class labels [B, num_classes].
            abs_step (int, optional): Global training step for controlling prototype updates.
            generate_cam (bool): If True, generate CAMs only (used in inference).

        Returns:
            dict or Tensor:
                - During training:
                    {
                        'cls_logits': Tensor,
                        'spatial_patch_logits': Tensor,
                        'temporal_patch_logits': Tensor,
                        'fusion_cam': Tensor,
                        'fusion_cam_refine': Tensor,
                        'proto_entropy': Tensor (if abs_step >= 4000),
                        'proto_dis': Tensor (if abs_step >= 4000)
                    }
                - During CAM generation: Tensor of shape [B, num_classes, H, W]
        """
        # Permute input to [B, T, H, W, C] -> [B, T, C, H, W]
        x = x.permute(0, 1, 4, 2, 3)  
        B, T, C, H, W = x.shape 
        # Extract day-of-year temporal information from the last channel 
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]   
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=365).to(torch.float32)
        xt = xt.reshape(-1, 365)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        # Patch embedding and temporal token integration
        z = self.to_patch_embedding(x)   
        z = z.reshape(B, -1, T, self.dim)
        z += temporal_pos_embedding.unsqueeze(1)
        z = z.reshape(-1, T, self.dim)  
        # Add temporal classification token 
        z_cls_t = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        z_in_t = torch.cat((z_cls_t, z), dim=1)   
        # Temporal transformer
        z_out_t, temporal_transformer_attention_weights = self.temporal_transformer(z_in_t)
        temporal_transformer_attention_weights = temporal_transformer_attention_weights.reshape(
            temporal_transformer_attention_weights.shape[0], B, 
            temporal_transformer_attention_weights.shape[1]//B,
            temporal_transformer_attention_weights.shape[2],
            temporal_transformer_attention_weights.shape[3])
        with torch.no_grad():
            temporal_to_class_attention = temporal_transformer_attention_weights[:,:,:,:self.num_classes,self.num_classes:].mean(0).mean(1) 
        z_seq_t = z_out_t[:, self.num_classes:].clone() 
        z_dense_t = z_out_t[:, :self.num_classes] 
        # Temporal patch logits
        temporal_patch = z_dense_t.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 3, 1).reshape(B*self.num_classes, self.dim, self.num_patches_1d, self.num_patches_1d)
        temporal_patch = temporal_patch.contiguous()
        temporal_feature = temporal_patch.clone()
        temporal_feature = rearrange(temporal_feature, "(b k) d h w -> b k d h w", b=B)
        temporal_patch = self.patch_head(temporal_patch) 
        temporal_patch = temporal_patch.reshape(B, self.num_classes, self.num_patches_1d, self.num_patches_1d)
        temporal_patch_logits = torch.mean(temporal_patch, dim=(2, 3)) 
        # Spatial patch logits     
        z_dense_t = z_dense_t.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        z_s = z_dense_t + self.space_pos_embedding  
        z_s = self.dropout(z_s)   
        z_cls_s = repeat(self.space_token, '() N d -> b N d', b=B * self.num_classes)
        z_in_s = torch.cat((z_cls_s, z_s), dim=1)
        z_out_s = self.space_transformer(z_in_s)
        z_global_s = z_out_s[:, 0]  
        cls_logits = self.mlp_head(z_global_s.reshape(-1, self.dim))
        cls_logits = cls_logits.reshape(B, self.num_classes) 
        z_dense_s = z_out_s[:, 1:] 
        n, p, c = z_dense_s.shape
        z_dense_s = torch.reshape(z_dense_s, [n, int(p ** 0.5), int(p ** 0.5), c]) 
        z_dense_s = z_dense_s.permute([0, 3, 1, 2]) 
        z_dense_s = z_dense_s.contiguous()
        z_dense_s = self.patch_head(z_dense_s) 
        z_dense_s= z_dense_s.reshape(B, self.num_classes, self.num_patches_1d, self.num_patches_1d)
        spatial_patch_logits = torch.mean(z_dense_s, dim=(2, 3)) 
        res = {
            'cls_logits':cls_logits,
            'spatial_patch_logits':spatial_patch_logits,
            'temporal_patch_logits':temporal_patch_logits
        }
        # Training phase: prototype learning + CAM refinement
        if self.training:
            if abs_step>=4000:
                cams = (F.relu(z_dense_s) + F.relu(temporal_patch)) / 2    
                cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5
                fg_proto_entropy, fg_proto_dis = self.prototype_learning(temporal_feature, cams, cls_label_gt, is_bg=False)
                bg_proto_entropy, bg_proto_dis = self.prototype_learning(temporal_feature, cams, cls_label_gt, is_bg=True)
                res['proto_entropy'] = fg_proto_entropy + bg_proto_entropy
                res['proto_dis'] = fg_proto_dis + bg_proto_dis
            temporal_to_class_attention = F.softmax(temporal_to_class_attention, dim=2).unsqueeze(2).unsqueeze(-1)
            z_seq_t = rearrange(z_seq_t, '(b h) t d -> b h t d', b=B).unsqueeze(1)
            weighted_tem = torch.sum(z_seq_t * temporal_to_class_attention, dim=3)
            weighted_tem = rearrange(weighted_tem, 'b k (h w) d -> b k d h w', h=self.num_patches_1d)
            fusion_cam = (F.relu(z_dense_s) + F.relu(temporal_patch)) / 2
            refine_cams = torch.zeros_like(fusion_cam)
            for k in range(cls_label_gt.shape[1]):
                if cls_label_gt[:,k].sum() == 0:
                    continue
                refine_cams[:,k] = self.taap_module.forward(weighted_tem[:,k], fusion_cam[:,k].clone().unsqueeze(1)).squeeze(1)
            res['fusion_cam'] = fusion_cam
            res['fusion_cam_refine'] = (refine_cams+fusion_cam)/2
            return res       
        # Inference: CAM generation using learned prototypes
        elif generate_cam:
            temporal_to_class_attention = F.softmax(temporal_to_class_attention, dim=2).unsqueeze(2).unsqueeze(-1)
            z_seq_t = rearrange(z_seq_t, '(b h) t d -> b h t d', b=B).unsqueeze(1)
            weighted_tem = torch.sum(z_seq_t * temporal_to_class_attention, dim=3)
            weighted_tem = rearrange(weighted_tem, 'b k (h w) d -> b k d h w', h=self.num_patches_1d)
            temporal_cam = torch.zeros(B*self.num_patches_1d*self.num_patches_1d, self.num_classes).cuda()
            self.fg_proto.data.copy_(l2_normalize(self.fg_proto))
            self.bg_proto.data.copy_(l2_normalize(self.bg_proto))
            for class_id in range(self.num_classes):
                fea_k = temporal_feature[:,class_id].clone()
                fea_k = rearrange(fea_k, 'b c h w -> (b h w) c')
                fea_k = l2_normalize(fea_k)
                cam_k = F.cosine_similarity(fea_k.unsqueeze(1), self.fg_proto[class_id].unsqueeze(0), dim=-1)
                cam_bg = F.cosine_similarity(fea_k.unsqueeze(1), self.bg_proto[class_id].unsqueeze(0), dim=-1)
                cam_k = torch.amax(cam_k, dim=1)
                cam_bg = torch.amax(cam_bg, dim=1)
                temporal_cam[:,class_id] = cam_k - cam_bg
            temporal_cam = rearrange(temporal_cam, "(b h w) k -> b k h w", b=B, h=self.num_patches_1d)
            temporal_cam = F.relu(temporal_cam)
            refine_cams = torch.zeros_like(temporal_cam)
            for k in range(cls_label_gt.shape[1]):
                if cls_label_gt[:,k].sum() == 0:
                    continue
                refine_cams[:,k] = self.taap_module.forward(weighted_tem[:,k], temporal_cam[:,k].clone().unsqueeze(1)).squeeze(1)            
            temporal_cam = (temporal_cam + refine_cams) / 2
            return temporal_cam
        # eval: return logits only
        else:
            return res
