# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from models.cdn import get_contrastive_denoising_training_group

from .mlp import MLP
from .ffn import FFN
from .backbone import BackboneWithPE
from .deformable_transformer import DeformableTransformer
from .query_updater import build as build_query_updater
from .utils import get_clones, pos_to_pos_embed

from .backbone import build as build_backbone_with_pe
from .deformable_transformer import build as build_deformable_transformer

from utils.nested_tensor import NestedTensor
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid

from torch.utils.checkpoint import checkpoint
import torch.nn.init as init


class HNCD(nn.Module):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer,
                 query_updater: nn.Module,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False, det_db=None,
                 num_cdn_group: int = 3, id_noise_ratio: float = 0.3,
                 box_noise_scale: float = 0.4, cdn_k: int = 3, use_tiny_noise: bool = True,
                 use_proposals: bool = True):
        super(HNCD, self).__init__()

        self.num_classes = num_classes
        self.n_det_queries = n_det_queries
        self.n_feature_levels = n_feature_levels
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize
        self.use_proposals = use_proposals
        
        # NOTE CDN
        self.num_cdn_group = num_cdn_group
        self.id_noise_ratio = id_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.cdn_k = cdn_k
        self.use_tiny_noise = use_tiny_noise
        if num_cdn_group > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])
        # NOTE proposal
        self.det_db = det_db
        if det_db is not None:
            self.position = nn.Embedding(n_det_queries, 4)
            self.yolox_embed = nn.Embedding(1, hidden_dim)
            self.query_embed = nn.Embedding(n_det_queries, hidden_dim)
        
        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 4))  # (N_det, 4)
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))       # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))   # (N_det, 2C)
        assert self.n_feature_levels > 1
        n_backbone_inter_layers = backbone.n_inter_layers()
        n_backbone_inter_channels = backbone.n_inter_channels()
        feature_proj_list = []
        for i in range(n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[i], out_channels=self.hidden_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        for _ in range(self.n_feature_levels - n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[-1], out_channels=self.hidden_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        self.feature_projs = nn.ModuleList(feature_proj_list)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.transformer.get_n_dec_layers())])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.transformer.get_n_dec_layers())])

    def pos2posemb(self, pos, num_pos_feats=64, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        posemb = pos[..., None] / dim_t
        posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
        return posemb
    
    def forward(self, frame: NestedTensor, tracks: list[TrackInstances], proposals=None, gt_target=None):
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/memotr/", exist_ok=True)

        # 图像经过 backbone
        if self.use_checkpoint and self.checkpoint_level != 3:
            features, pos = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            features, pos = self.backbone(frame)

        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.feature_projs[layer](src))
            masks.append(mask)
        if self.n_feature_levels > len(srcs):
            srcs_len = len(srcs)
            for layer in range(srcs_len, self.n_feature_levels):
                if layer == srcs_len:
                    src = self.feature_projs[layer](features[-1].tensors)
                else:
                    src = self.feature_projs[layer](srcs[-1])
                mask = frame.masks
                mask = F.interpolate(mask[None, ...].float(), size=src.shape[-2:])[0].to(torch.bool)
                pos.append(self.backbone.position_embedding(NestedTensor(src, mask)).to(src.device))
                srcs.append(src)
                masks.append(mask)
        # srcs is n_feature_levels * [(B, C, H, W)]
        # masks is n_feature_levels * [(B, H, W)]
        # pos is n_features_levels * [(B, C, H, W)]
        proposals = proposals[0] if proposals is not None else None
        num_proposals = len(proposals) if self.use_proposals and proposals is not None else 0
        reference_points = self.get_reference_points(tracks=tracks, proposals=proposals).to(srcs[0].device)      # (B, Nd+Nq, 2/4)
        query_embed = self.get_query_embed(tracks=tracks, proposals=proposals).to(srcs[0].device)
        query_mask_without_cdn = self.get_query_mask(tracks=tracks, proposals=proposals).to(srcs[0].device)                  # (B, Nd+Nq)

        # NOTE CDN
        # CDN tokens are inserted INTO the middle of the sequence, between
        # (det + proposals) and (track), instead of being prepended to the front.
        # This keeps the decoder's `:n_det_queries` slice (used at the merge_det_track
        # boundary in deformable_decoder.py) correctly pointing at det queries; if
        # CDN were at the front the slice would shift by num_cdn and freeze the tail
        # of the det block at layer 0, breaking train/eval consistency.
        det_query_ori = query_embed[0][:self.n_det_queries + num_proposals]
        attn_mask = None
        cdn_meta = None
        num_cdn = 0
        query_mask = query_mask_without_cdn

        if self.training and self.num_cdn_group > 0:
            targets = {
                    'boxes': gt_target['boxes'],
                    'labels': gt_target['labels']
                }
            n_left = self.n_det_queries + num_proposals
            n_right = query_embed.shape[1] - n_left  # length of the track suffix
            denoising_logits, denoising_bbox_unact, attn_mask, cdn_meta = \
                get_contrastive_denoising_training_group([targets],
                                                        self.num_classes,
                                                        n_left,
                                                        n_right,
                                                        self.denoising_class_embed,
                                                        self.num_cdn_group,
                                                        self.id_noise_ratio,
                                                        self.box_noise_scale,
                                                        self.cdn_k,
                                                        self.use_tiny_noise
                                                    )
            if denoising_bbox_unact is not None:
                num_cdn = denoising_logits.shape[1] # 获取cdn数量
                denoising_bbox_unact = denoising_bbox_unact.to(self.denoising_class_embed.weight.device)
                attn_mask = attn_mask.to(self.denoising_class_embed.weight.device)
                # Insert CDN block between [det+proposals] (left) and [track] (right).
                qe_left = query_embed[:, :n_left, :]
                qe_right = query_embed[:, n_left:, :]
                query_embed = torch.cat([qe_left, denoising_logits, qe_right], dim=1)

                rp_left = reference_points[:, :n_left, :]
                rp_right = reference_points[:, n_left:, :]
                reference_points = torch.cat([rp_left, denoising_bbox_unact, rp_right], dim=1)

                cdn_mask_part = torch.zeros(query_embed.shape[0], num_cdn,
                                            dtype=torch.bool, device=query_mask_without_cdn.device)
                qm_left = query_mask_without_cdn[:, :n_left]
                qm_right = query_mask_without_cdn[:, n_left:]
                query_mask = torch.cat([qm_left, cdn_mask_part, qm_right], dim=1)
        
        # DETR:
        outputs, init_reference, inter_references, inter_queries = self.transformer(
            srcs=srcs,
            masks=masks,
            pos_embeds=pos,
            query_embed=query_embed,
            ref_pts=reference_points,
            query_mask=query_mask,
            num_proposals=num_proposals,
            num_cdn=num_cdn,
            attn_mask=attn_mask
        )
        # outputs: (n_dec_layers, B, Nd+Nq, C)
        # init_reference: (B, Nd+Nq, 2)
        # inter_references: (n_dec_layers, B, Nd+Nq, 4)
        output_classes, output_bboxes = [], []
        assert outputs.ndim == 4, f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C, " \
                                  f"but get n_dim={outputs.ndim}"
        for level in range(outputs.shape[0]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            if reference.shape[-1] == 4:
                bbox_tmp += reference
            else:
                assert reference.shape[-1] == 2, f"Reference should have only 2 coord, but get {reference.shape[-1]}."
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)

            if self.visualize:
                
                torch.save(reference[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/hncd/detection_ref_pts_layer_{level}.tensor")
                torch.save(reference[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/hncd/track_ref_pts_layer_{level}.tensor")
                torch.save(output_class[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/hncd/detection_logits_layer_{level}.tensor")
                torch.save(output_class[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/hncd/track_logits_layer_{level}.tensor")
                torch.save(output_bbox[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/hncd/detection_boxes_layer_{level}.tensor")
                torch.save(output_bbox[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/hncd/track_boxes_layer_{level}.tensor")

        output_classes = torch.stack(output_classes, dim=0) # [B, 6, N_dn+N_det+N_track, 1]
        output_bboxes = torch.stack(output_bboxes, dim=0) # [B, 6, N_dn+N_det+N_track, 4]

        # 分离CDN
        # The decoder output sequence is [Q_left | CDN | Q_right]. We split into three
        # parts, save the middle as cdn_*, and re-cat left+right back into the regular
        # outputs (which the rest of the model and criterion expect to be CDN-free).
        if self.training and self.num_cdn_group > 0:
            nl = cdn_meta['cdn_num_left']
            nc = cdn_meta['cdn_num_denoising']
            nr = cdn_meta['cdn_num_right']
            split_q = [nl, nc, nr]

            obb_l, cdn_output_bboxes, obb_r = torch.split(output_bboxes, split_q, dim=2)
            output_bboxes = torch.cat([obb_l, obb_r], dim=2)

            oc_l, cdn_output_classes, oc_r = torch.split(output_classes, split_q, dim=2)
            output_classes = torch.cat([oc_l, oc_r], dim=2)

            ir_l, cdn_inter_references, ir_r = torch.split(inter_references, split_q, dim=2)
            inter_references = torch.cat([ir_l, ir_r], dim=2)

            iq_l, cdn_inter_queries, iq_r = torch.split(inter_queries, split_q, dim=2)
            inter_queries = torch.cat([iq_l, iq_r], dim=2)

            op_l, cdn_outputs, op_r = torch.split(outputs, split_q, dim=2)
            outputs = torch.cat([op_l, op_r], dim=2)

            qm_l, cdn_query_mask, qm_r = torch.split(query_mask, split_q, dim=1)
            query_mask = torch.cat([qm_l, qm_r], dim=1)

            # init_reference also has CDN inserted in the middle; strip it so that
            # res["init_ref_pts"] is indexable by [det+proposals+track] positions
            # (the criterion uses this directly with unmatched_indexes).
            ir0_l, _, ir0_r = torch.split(init_reference, split_q, dim=1)
            init_reference = torch.cat([ir0_l, ir0_r], dim=1)
        
        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]) if self.use_dab       # (B, Nd+Nq, 4)
            else inverse_sigmoid(inter_references[-2, :, :, :]),                                 # (B, Nd+Nq, 2)
            "query_mask": query_mask,                   # (B, Nd+Nq)
            "det_query_embed": det_query_ori,
            "init_ref_pts": inverse_sigmoid(init_reference)
        }
        if self.aux_loss:
            res["aux_outputs"] = self.set_aux_loss(output_classes=output_classes,
                                                   output_bboxes=output_bboxes,
                                                   query_mask=query_mask,
                                                   queries=inter_queries)
        if self.training and self.num_cdn_group > 0 and denoising_bbox_unact is not None:
            res['cdn_outputs'] = self.set_cdn_loss(output_classes=cdn_output_classes,
                                                  output_bboxes=cdn_output_bboxes,
                                                  query_mask=cdn_mask_part,
                                                  queries=cdn_inter_queries)
            res['cdn_meta'] = cdn_meta
        
        res["outputs"] = outputs[-1]     # (B, Nd+Nq, C)
        return res

    @torch.jit.unused
    def set_aux_loss(self, output_classes, output_bboxes, query_mask, queries):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]
    
    @torch.jit.unused
    def set_cdn_loss(self, output_classes, output_bboxes, query_mask, queries):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes, output_bboxes, queries)
        ]

    # def get_det_reference_points(self) -> torch.Tensor:
    #     """
    #     Returns: (Nd, 2)
    def get_det_reference_points(self):
        """
        Returns: (Nd, 2/4)
        """
        if self.use_dab:
            return self.det_anchor
        else:
            return self.transformer.reference_points(self.det_query_embed[:, :self.hidden_dim])

    def get_track_reference_points(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2/4)
        """
        max_len = max([len(t.ref_pts) for t in tracks])
        if self.use_dab:
            references = torch.zeros((len(tracks), max_len, 4))
        else:
            # references = torch.zeros((len(tracks), max_len, 2))
            references = torch.zeros((len(tracks), max_len, 4))
        for i in range(len(tracks)):
            references[i, :len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: list[TrackInstances]):
        """
        Returns: (B, Nq, 2C)
        """
        max_len = max([len(t.query_embed) for t in tracks])
        if self.use_dab:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim))
        else:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, :len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: list[TrackInstances], proposals=None):
        # When use_proposals=True we MUST always go through the (position, query_embed)
        # branch — even if `proposals` happens to be empty for a particular frame —
        # otherwise the set of trainable parameters used per iteration is not constant,
        # and DDP `find_unused_parameters=True` raises
        # "Expected to mark a variable ready only once" once a frame with empty
        # proposals is sampled (this matches MOTRv2's behaviour, see
        # MOTRv2/models/motr.py:_generate_empty_tracks).
        if self.use_proposals:
            if proposals is not None and len(proposals) > 0:
                proposals = inverse_sigmoid(proposals).to(self.position.weight.device)
                det_references = torch.cat([self.position.weight, proposals[:, :4]]).unsqueeze(0)
            else:
                det_references = self.position.weight.unsqueeze(0)
        else:
            det_references = self.get_det_reference_points().repeat(len(tracks), 1, 1)                      # (B, Nd, 2)
            if det_references.shape[-1] == 2:
                det_references = torch.cat(
                    (det_references, torch.zeros_like(det_references, device=det_references.device)),
                    dim=-1
                )
        track_references = self.get_track_reference_points(tracks=tracks).to(det_references.device)     # (B, Nq, 2)
        return torch.cat((det_references, track_references), dim=1)

    def get_query_embed(self, tracks: list[TrackInstances], proposals=None):
        """
        Returns: (B, Nd+Nq, 2C)
        """
        if self.use_dab:
            # Always pick the (query_embed, yolox_embed) branch when use_proposals=True,
            # for the same DDP-stability reason as in get_reference_points above.
            if self.use_proposals:
                if proposals is not None and len(proposals) > 0:
                    det_query_embed = torch.cat([self.query_embed.weight, self.pos2posemb(proposals[:, 4:], self.hidden_dim).to(self.yolox_embed.weight.device) + self.yolox_embed.weight]).unsqueeze(0)
                else:
                    det_query_embed = self.query_embed.weight.unsqueeze(0)
            else:
                det_query_embed = self.det_query_embed
                det_query_embed = det_query_embed.repeat(len(tracks), 1, 1)
        else:
            det_query_embed = self.det_query_embed.repeat(len(tracks), 1, 1)                    # (B, Nd, 2C)
        track_query_embed = self.get_track_query_embed(tracks).to(det_query_embed.device)       # (B, Nq, 2C)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: list[TrackInstances], proposals = None):
        """
        Returns: (B, Nd+Nq)
        """
        track_max_len = max([len(t.query_embed) for t in tracks])
        num_proposals = len(proposals) if self.use_proposals and proposals is not None else 0
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries + num_proposals)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask), dim=1).to(proposals.device)

    def postprocess_single_frame(self, previous_tracks: List[TrackInstances],
                                 new_tracks: List[TrackInstances],
                                 unmatched_dets: List[TrackInstances] | None,
                                 no_augment: bool = False):
        """
        Query updating.
        """
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


def build(config: dict):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
    }
    assert config["DATASET"] in dataset_num_classes, f"Do not know the class num of {config['DATASET']} dataset."
    num_classes = dataset_num_classes[config["DATASET"]]

    backbone_with_pe = build_backbone_with_pe(config=config)
    deformable_transformer = build_deformable_transformer(config=config)
    # Architecture switch: memotr (TIM, default) or motrv2 (QIM).
    arch = str(config.get("ARCH", "memotr")).lower()
    if arch == "motrv2":
        from .qim import build as build_qim
        query_updater = build_qim(config=config)
    elif arch == "memotr":
        query_updater = build_query_updater(config=config)
    else:
        raise ValueError(f"Unknown ARCH '{arch}'. Expected one of: memotr, motrv2.")
    return HNCD(
        backbone=backbone_with_pe,
        transformer=deformable_transformer,
        query_updater=query_updater,
        num_classes=num_classes,
        n_det_queries=config["NUM_DET_QUERIES"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        aux_loss=True,
        with_box_refine=True,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"],
        det_db=config["DET_DB"],
        num_cdn_group=config["NUM_CDN_GROUP"],
        id_noise_ratio=config["ID_NOISE_RATIO"],
        box_noise_scale=config["BOX_NOISE_SCALE"],
        cdn_k=config["CDN_K"],
        use_tiny_noise=config['USE_TINY_NOISE'],
        use_proposals=config.get("USE_PROPOSALS", True)
    )
