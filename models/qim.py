# MOTRv2-style Query Interaction Module v2 ported to MeMOTR's TrackInstances API.
#
# Differences from MeMOTR's TIM (`query_updater.py`):
#   * No long-term memory.  TIM keeps a running EMA `long_memory` and a `last_output`
#     and runs a cross-attention between {short_memory, long_memory}; QIMv2 uses only
#     the current track set with a single self-attention pass.
#   * `update_query_pos` is False by default (we only update the query feature, not
#     the positional embedding) -- this matches MOTRv2's default config.
#
# Field name mapping  MOTRv2 → MeMOTR:
#     track_instances.output_embedding   ->  tracks.output_embed
#     track_instances.query_pos          ->  tracks.query_embed   (DAB mode: shape=hidden_dim)
#     track_instances.pred_boxes         ->  tracks.boxes
#     track_instances.obj_idxes          ->  tracks.ids
#     track_instances.scores             ->  max(sigmoid(tracks.logits))
#     track_instances.iou                ->  tracks.iou           (same name)
#     track_instances.ref_pts            ->  tracks.ref_pts       (NOTE: MOTRv2 stores this
#                                                                  in [0,1] sigmoid space;
#                                                                  MeMOTR stores in
#                                                                  inverse_sigmoid logit space.
#                                                                  We follow MeMOTR's convention
#                                                                  and apply .sigmoid() before
#                                                                  pos2posemb.)
import math
import torch
import torch.nn as nn

from typing import List

from .utils import pos_to_pos_embed, logits_to_scores
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid


class QueryInteractionModulev2(nn.Module):
    """MOTRv2-style track update on top of MeMOTR's TrackInstances API.

    Drop-in replacement for `models.query_updater.QueryUpdater` (TIM): same
    forward signature, returns the same kind of List[TrackInstances]. Only the
    internal track-feature update differs.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float,
                 use_dab: bool, update_threshold: float = 0.5,
                 tp_drop_ratio: float = 0.0, fp_insert_ratio: float = 0.0,
                 update_query_pos: bool = False, visualize: bool = False):
        super().__init__()
        # The current MeMOTR-base codebase only exercises use_dab=True (DAB-Deformable-DETR).
        # MOTRv2 itself also uses DAB-style 4D anchors (see MOTRv2/models/motr.py:393), so
        # we only support the DAB path here.
        assert use_dab, "QIMv2 in this codebase only supports use_dab=True"
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.use_dab = use_dab
        self.update_threshold = update_threshold
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.update_query_pos = update_query_pos
        self.visualize = visualize

        # --- Self-attention block (mirrors QueryInteractionModulev2._build_layers) ---
        self.self_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.dropout_inter = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)

        if update_query_pos:
            self.linear_pos1 = nn.Linear(hidden_dim, ffn_dim)
            self.linear_pos2 = nn.Linear(ffn_dim, hidden_dim)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(hidden_dim)

        self.linear_feat1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear_feat2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        if update_query_pos:
            self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Public forward — matches QueryUpdater (TIM) signature exactly.
    # ------------------------------------------------------------------
    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False):
        tracks = self._select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self._update_track_embedding(tracks)
        return tracks

    # ------------------------------------------------------------------
    # Internal: which tracks are kept and combined for the next frame.
    # MOTRv2 QIMv2 train: (obj_idxes >= 0) | (scores > 0.5)  +  iou<=0.5 -> obj_idx=-1
    # MOTRv2 QIMv2 eval : obj_idxes >= 0
    # We additionally honor TP-drop / FP-insert augmentation flags so that the
    # MeMOTR-side training augmentation pipeline is preserved.
    # ------------------------------------------------------------------
    def _select_active_tracks(self,
                              previous_tracks: List[TrackInstances],
                              new_tracks: List[TrackInstances],
                              unmatched_dets: List[TrackInstances] | None,
                              no_augment: bool = False):
        from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union

        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                # Combine all candidates: matched-from-previous + newly matched + unmatched dets.
                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = TrackInstances.cat_tracked_instances(active_tracks, unmatched_dets[b])
                    scores = torch.max(logits_to_scores(active_tracks.logits), dim=1).values
                    keep_idxes = (scores > self.update_threshold) | (active_tracks.ids >= 0)
                    active_tracks = active_tracks[keep_idxes]
                    # MOTRv2 QIMv2: tracks whose IoU vs GT is bad lose their object id.
                    active_tracks.ids[active_tracks.iou < 0.5] = -1
                else:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = active_tracks[(active_tracks.iou > 0.5) & (active_tracks.ids >= 0)]
                    if self.tp_drop_ratio > 0.0 and not no_augment:
                        if len(active_tracks) > 0:
                            tp_keep_idx = torch.rand((len(active_tracks),)) > self.tp_drop_ratio
                            active_tracks = active_tracks[tp_keep_idx]
                    if self.fp_insert_ratio > 0.0 and not no_augment:
                        selected_active_tracks = active_tracks[
                            torch.bernoulli(
                                torch.ones((len(active_tracks),)) * self.fp_insert_ratio
                            ).bool()
                        ]
                        if len(unmatched_dets[b]) > 0 and len(selected_active_tracks) > 0:
                            fp_num = len(selected_active_tracks)
                            if fp_num >= len(unmatched_dets[b]):
                                insert_fp = unmatched_dets[b]
                            else:
                                selected_active_boxes = box_cxcywh_to_xyxy(selected_active_tracks.boxes)
                                unmatched_boxes = box_cxcywh_to_xyxy(unmatched_dets[b].boxes)
                                iou, _ = box_iou_union(unmatched_boxes, selected_active_boxes)
                                fp_idx = torch.max(iou, dim=0).indices
                                fp_idx = torch.unique(fp_idx)
                                insert_fp = unmatched_dets[b][fp_idx]
                            active_tracks = TrackInstances.cat_tracked_instances(active_tracks, insert_fp)

                # If nothing remains, push a single dummy entry so DDP gradient sync works.
                if len(active_tracks) == 0:
                    device = next(self.parameters()).device
                    fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0,
                                                 hidden_dim=self.hidden_dim).to(device=device)
                    fake_tracks.query_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.output_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.ids = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.matched_idx = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.boxes = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.logits = torch.randn((1, active_tracks.logits.shape[1]),
                                                     dtype=torch.float, device=device)
                    fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
                    active_tracks = fake_tracks
                tracks.append(active_tracks)
        else:
            # Inference: B==1, just keep tracks with valid object ids.
            assert len(previous_tracks) == 1 and len(new_tracks) == 1
            active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[0], new_tracks[0])
            active_tracks = active_tracks[active_tracks.ids >= 0]
            tracks.append(active_tracks)

        return tracks

    # ------------------------------------------------------------------
    # Internal: refresh ref_pts from current boxes, then run a single
    # self-attention + FFN pass over the kept tracks to produce the next
    # frame's query content.
    # ------------------------------------------------------------------
    def _update_track_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            if len(tracks[b]) == 0:
                continue
            scores = torch.max(logits_to_scores(tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold

            # Update reference points from the latest predicted boxes for tracks
            # that are confident enough. Stored in MeMOTR's inverse_sigmoid logit space.
            if is_pos.any():
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(
                    tracks[b][is_pos].boxes.detach().clone()
                )

            # Build the QIMv2 inputs. ref_pts in this codebase lives in inverse_sigmoid
            # space, so apply .sigmoid() before the positional encoding to match
            # MOTRv2's convention (its position is stored in [0,1] directly).
            #
            # num_pos_feats is chosen so that the positional encoding output matches
            # hidden_dim exactly (no projection needed): pos_to_pos_embed produces
            # `ref_pts_dim * num_pos_feats` features, so we want
            #   num_pos_feats = hidden_dim // ref_pts_dim.
            # For DAB 4D anchors with hidden_dim=256 this gives 64, matching MOTRv2's
            # default `pos2posemb(num_pos_feats=64)` exactly.
            out_embed = tracks[b].output_embed                                         # (N, C)
            query_feat = tracks[b].query_embed                                         # (N, C)  -- DAB only
            ref_pts_dim = tracks[b].ref_pts.shape[-1]
            assert self.hidden_dim % ref_pts_dim == 0, \
                f"hidden_dim ({self.hidden_dim}) must be divisible by ref_pts_dim ({ref_pts_dim}) for QIMv2"
            query_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(),
                                         num_pos_feats=self.hidden_dim // ref_pts_dim)  # (N, C)

            q = k = query_pos + out_embed
            tgt = out_embed
            # nn.MultiheadAttention with batch_first=False expects (L, N, E); add fake batch dim.
            tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            tgt2 = self.linear2(self.dropout_inter(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # Optional positional update branch (matches MOTRv2 args.update_query_pos).
            if self.update_query_pos:
                query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
                query_pos = query_pos + self.dropout_pos2(query_pos2)
                query_pos = self.norm_pos(query_pos)
                # Note: MOTRv2 mutates query_pos here. In MeMOTR's DAB API the
                # positional encoding is recomputed from ref_pts inside the decoder, so
                # there is no separate place to store this updated query_pos.  Skipping
                # the in-place write keeps the data flow consistent (the FFN below still
                # picks up the refined `tgt`).

            # Update query feature (the part stored on the track for next frame's decoder).
            query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
            query_feat = query_feat + self.dropout_feat2(query_feat2)
            query_feat = self.norm_feat(query_feat)
            tracks[b].query_embed[is_pos] = query_feat[is_pos]

        return tracks


def build(config: dict):
    return QueryInteractionModulev2(
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        use_dab=config["USE_DAB"],
        update_threshold=config.get("UPDATE_THRESH", 0.5),
        tp_drop_ratio=config.get("TP_DROP_RATE", 0.0),
        fp_insert_ratio=config.get("FP_INSERT_RATE", 0.0),
        update_query_pos=config.get("UPDATE_QUERY_POS", False),
        visualize=config.get("VISUALIZE", False),
    )
