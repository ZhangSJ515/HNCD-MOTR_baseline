import torch
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.utils import inverse_sigmoid


def _canonicalize_negative_source(source: str, use_tiny_noise: bool) -> str:
    """Resolve the hard-negative source with backward-compatible fallback."""
    if source is None:
        return "knn" if use_tiny_noise else "nearest"

    source = str(source).strip().lower()
    alias_map = {
        "knn": "knn",
        "knn_random": "knn",
        "knn_random_sampling": "knn",
        "nearest": "nearest",
        "single_nearest": "nearest",
        "random_gt": "random_gt",
        "random": "random_gt",
        "random_sampling": "random_gt",
    }
    if source not in alias_map:
        raise ValueError(f"Unsupported CDN_NEGATIVE_SOURCE '{source}'.")
    return alias_map[source]


def _canonicalize_replaced_noise_mode(mode: str, use_tiny_noise: bool) -> str:
    """Resolve the replaced-negative noise strategy with backward-compatible fallback."""
    if mode is None:
        return "reduced" if use_tiny_noise else "none"

    mode = str(mode).strip().lower()
    alias_map = {
        "reduced": "reduced",
        "tiny": "reduced",
        "tiny_noise": "reduced",
        "standard": "standard",
        "full": "standard",
        "standard_noise": "standard",
        "none": "none",
        "no_noise": "none",
    }
    if mode not in alias_map:
        raise ValueError(f"Unsupported CDN_REPLACED_NOISE_MODE '{mode}'.")
    return alias_map[mode]


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries_left,
    num_queries_right,
    class_embed,
    num_cdn_group=3,
    id_noise_ratio=0.3,
    box_noise_scale=0.4,
    cdn_k=5,
    use_tiny_noise=False,
    cdn_negative_source=None,
    cdn_replaced_noise_mode=None,
    cdn_replaced_noise_scale=None
):
    """
    Unified Contrastive Denoising (CDN) / Hard Negative Denoising (HND) Group Generator.

    The CDN block is intended to be inserted INTO the middle of the query sequence,
    between the (det + proposals) prefix and the (track) suffix. The returned attn_mask
    therefore covers the full sequence:
        [Q_left (det+proposals) | CDN groups | Q_right (track)]
    of length num_queries_left + num_denoising + num_queries_right.

    Why "in the middle" and not "at the front":
        - The decoder freezes track ref_pts at the merge_det_track layer via the
          slice ``[:, :n_det_queries, :]`` (refine prefix, keep suffix). With CDN at
          the front this slice would refine "CDN + first portion of det" and freeze
          "the tail of det + tracks", causing a train/eval mismatch (CDN absent at
          eval shifts the boundary). Inserting CDN between det+proposals and track
          keeps the prefix ":n_det_queries" pointing at det, and pushes the rest
          (including CDN) into the frozen tail at that layer only.
        - At inference time CDN is never injected, so the sequence reduces to
          [det+proposals | track], identical to the original layout. Existing
          checkpoints stay valid for inference.

    Args:
        num_queries_left (int): number of queries that come BEFORE the CDN block in the
            final sequence (det queries + proposal queries).
        num_queries_right (int): number of queries that come AFTER the CDN block (track
            queries).
        cdn_k (int): The 'k' for K-NN sampling when the negative source is K-NN.
        use_tiny_noise (bool): Legacy fallback flag retained for backward compatibility.
        cdn_negative_source (str): One of {"knn", "nearest", "random_gt"}.
        cdn_replaced_noise_mode (str): One of {"reduced", "standard", "none"}.
        cdn_replaced_noise_scale (float): Explicit hard-negative noise scale used when
            cdn_replaced_noise_mode == "reduced". If None, falls back to the legacy
            0.1 * box_noise_scale behavior.
    """
    if num_cdn_group <= 0:
        return None, None, None, None

    negative_source = _canonicalize_negative_source(cdn_negative_source, use_tiny_noise)
    replaced_noise_mode = _canonicalize_replaced_noise_mode(cdn_replaced_noise_mode, use_tiny_noise)
    if cdn_replaced_noise_scale is None:
        replaced_noise_scale = box_noise_scale * 0.1
    else:
        replaced_noise_scale = float(cdn_replaced_noise_scale)

    # --- Section 1, 2, 3: Data Preparation and Masking ---
    num_group = num_cdn_group
    num_gts = [len(t["labels"]) for t in targets]
    # Get device from class_embed to ensure consistency, especially when targets have empty tensors on CPU
    device = next(class_embed.parameters()).device
    bs = len(num_gts)

    max_gt_num = max(num_gts) if num_gts else 0
    if max_gt_num == 0:
        # NOTE 构造假数据以确保DDP梯度同步
        padding_idx = num_classes  # Use num_classes as padding index for background
        num_fake_queries = 2

        fake_query_class = torch.full(
            (bs, num_fake_queries), padding_idx, dtype=torch.long, device=device
        )

        fake_query_bbox = torch.tensor(
            [[0.5, 0.5, 0.1, 0.1]], device=device
        ).repeat(bs, num_fake_queries, 1)

        # 生成伪造的 logits 和 bbox_unact
        input_query_logits = class_embed(fake_query_class)

        fake_query_bbox = torch.clamp(fake_query_bbox, 1e-6, 1.0 - 1e-6)
        input_query_bbox_unact = inverse_sigmoid(fake_query_bbox)

        # 生成注意力掩码 (CDN-in-middle layout)
        tgt_size = num_queries_left + num_fake_queries + num_queries_right
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        cdn_start = num_queries_left
        cdn_end = num_queries_left + num_fake_queries
        # Q_left and Q_right cannot see fake CDN queries
        attn_mask[:cdn_start, cdn_start:cdn_end] = True
        attn_mask[cdn_end:, cdn_start:cdn_end] = True
        # Fake CDN cannot see Q_left/Q_right either
        attn_mask[cdn_start:cdn_end, :cdn_start] = True
        attn_mask[cdn_start:cdn_end, cdn_end:] = True

        # 即使是假数据，也需要一个符合格式的dn_meta
        dn_positive_idx = [torch.empty(0, dtype=torch.long, device=device) for _ in range(bs)]
        dn_meta = {
            "cdn_positive_idx": dn_positive_idx,
            "cdn_num_group": num_group,  # Can keep the group num
            "cdn_num_left": num_queries_left,
            "cdn_num_denoising": num_fake_queries,
            "cdn_num_right": num_queries_right,
        }
        return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta

    input_query_class_list, input_query_bbox_list, pad_gt_mask_list = [], [], []
    for i in range(bs):
        num_gt = num_gts[i]
        cls_tensor = torch.full([max_gt_num], num_classes, dtype=torch.long, device=device)
        box_tensor = torch.zeros([max_gt_num, 4], device=device)
        mask_tensor = torch.zeros([max_gt_num], dtype=torch.bool, device=device)
        if num_gt > 0:
            cls_tensor[:num_gt] = targets[i]["labels"]
            box_tensor[:num_gt] = targets[i]["boxes"]
            mask_tensor[:num_gt] = True
        input_query_class_list.append(cls_tensor)
        input_query_bbox_list.append(box_tensor)
        pad_gt_mask_list.append(mask_tensor)

    input_query_class = torch.stack(input_query_class_list)
    input_query_bbox_padded = torch.stack(input_query_bbox_list)
    pad_gt_mask = torch.stack(pad_gt_mask_list)

    num_denoising = int(max_gt_num * 2 * num_group)
    input_query_class = input_query_class.repeat(1, 2 * num_group)
    input_query_bbox = input_query_bbox_padded.repeat(1, 2 * num_group, 1)
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)

    base_neg_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    base_neg_mask[:, max_gt_num:] = 1
    neg_mask = base_neg_mask.repeat(1, num_group, 1).bool().squeeze(-1)

    positive_gt_mask = (~neg_mask) & pad_gt_mask
    dn_positive_idx_flat = torch.nonzero(positive_gt_mask)
    dn_positive_idx = []
    for b in range(bs):
        idx_in_batch = dn_positive_idx_flat[dn_positive_idx_flat[:, 0] == b, 1]
        dn_positive_idx.append(idx_in_batch)

    # --- Section 4: Hard Negative Generation ---
    replace_mask = (
        (torch.rand(bs, num_denoising, device=device) < id_noise_ratio)
        & neg_mask
        & pad_gt_mask
    )

    for b in range(bs):
        num_gt = num_gts[b]
        if num_gt <= 1:
            continue

        current_gt_boxes = targets[b]["boxes"].to(device)
        gt_centers = current_gt_boxes[:, :2]
        pairwise_dist = torch.cdist(gt_centers, gt_centers, p=2.0)
        pairwise_dist.fill_diagonal_(float("inf"))

        if negative_source == "knn":
            k_for_topk = min(cdn_k, num_gt - 1)
            if k_for_topk == 0:
                continue
            _, topk_indices = torch.topk(pairwise_dist, k=k_for_topk, dim=1, largest=False)
            random_choice = torch.randint(k_for_topk, size=(num_gt,), device=device)
            nearest_gt_indices = topk_indices[torch.arange(num_gt, device=device), random_choice]
        elif negative_source == "nearest":
            nearest_gt_indices = torch.argmin(pairwise_dist, dim=1)
        elif negative_source == "random_gt":
            random_choice = torch.randint(num_gt - 1, size=(num_gt,), device=device)
            gt_indices = torch.arange(num_gt, device=device)
            nearest_gt_indices = random_choice + (random_choice >= gt_indices).long()
        else:
            raise ValueError(f"Unsupported negative source '{negative_source}'.")

        to_replace_indices_batch = torch.nonzero(replace_mask[b]).squeeze(-1)
        for neg_idx_tiled in to_replace_indices_batch:
            orig_gt_idx = neg_idx_tiled % max_gt_num
            if orig_gt_idx >= num_gt:
                continue
            nearest_idx_orig = nearest_gt_indices[orig_gt_idx]
            nearest_gt_box = current_gt_boxes[nearest_idx_orig]
            input_query_bbox[b, neg_idx_tiled] = nearest_gt_box

    # --- Section 5: Noise Injection ---
    if box_noise_scale > 0:
        known_bbox_xyxy = box_cxcywh_to_xyxy(input_query_bbox)
        whwh = torch.cat([input_query_bbox[..., 2:], input_query_bbox[..., 2:]], dim=-1)
        diff = whwh * 0.5
        rand_sign = (torch.randint_like(input_query_bbox, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0)
        rand_part = torch.rand_like(input_query_bbox)

        positive_mask = (~neg_mask) & pad_gt_mask
        replaced_neg_mask = replace_mask & pad_gt_mask
        non_replaced_neg_mask = neg_mask & (~replace_mask) & pad_gt_mask

        noise_application_mask = positive_mask | non_replaced_neg_mask
        noise_scale_factor = torch.zeros_like(input_query_bbox)
        noise_scale_factor = torch.where(
            noise_application_mask.unsqueeze(-1),
            torch.full_like(noise_scale_factor, box_noise_scale),
            noise_scale_factor
        )

        shifted_rand_part = rand_part.clone()
        negative_standard_mask = non_replaced_neg_mask

        if replaced_noise_mode == "reduced":
            noise_application_mask = noise_application_mask | replaced_neg_mask
            noise_scale_factor = torch.where(
                replaced_neg_mask.unsqueeze(-1),
                torch.full_like(noise_scale_factor, replaced_noise_scale),
                noise_scale_factor
            )
        elif replaced_noise_mode == "standard":
            noise_application_mask = noise_application_mask | replaced_neg_mask
            noise_scale_factor = torch.where(
                replaced_neg_mask.unsqueeze(-1),
                torch.full_like(noise_scale_factor, box_noise_scale),
                noise_scale_factor
            )
            negative_standard_mask = negative_standard_mask | replaced_neg_mask
        elif replaced_noise_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported replaced noise mode '{replaced_noise_mode}'.")

        shifted_rand_part = torch.where(
            negative_standard_mask.unsqueeze(-1),
            shifted_rand_part + 1.0,
            shifted_rand_part
        )
        noise = shifted_rand_part * rand_sign * diff
        final_noise = noise * noise_scale_factor
        final_bbox_xyxy = torch.where(
            noise_application_mask.unsqueeze(-1),
            known_bbox_xyxy + final_noise,
            known_bbox_xyxy,
        )

        # Common post-processing for both versions
        final_bbox_xyxy = torch.clamp(final_bbox_xyxy, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(final_bbox_xyxy)
        input_query_bbox = torch.clamp(input_query_bbox, 1e-6, 1.0 - 1e-6)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
    else:
        input_query_bbox_unact = inverse_sigmoid(torch.clamp(input_query_bbox, 1e-6, 1.0 - 1e-6))

    # --- Section 6, 7, 8: Logits, Attn Mask, Meta Info ---
    # CDN-in-middle layout: [Q_left (num_queries_left) | CDN (num_denoising) | Q_right (num_queries_right)]
    input_query_logits = class_embed(input_query_class.to(class_embed.weight.device))
    tgt_size = num_queries_left + num_denoising + num_queries_right
    cdn_start = num_queries_left
    cdn_end = num_queries_left + num_denoising
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # Q_left and Q_right cannot see CDN queries (CDN must not leak into the regular forward pass)
    attn_mask[:cdn_start, cdn_start:cdn_end] = True
    attn_mask[cdn_end:, cdn_start:cdn_end] = True
    # Each CDN group can only attend to itself: block all cols outside its own block
    for i in range(num_group):
        group_start = cdn_start + max_gt_num * 2 * i
        group_end = min(cdn_start + max_gt_num * 2 * (i + 1), cdn_end)
        if group_start >= group_end:
            continue
        if group_start > 0:
            attn_mask[group_start:group_end, :group_start] = True
        if group_end < tgt_size:
            attn_mask[group_start:group_end, group_end:] = True

    dn_meta = {
        "cdn_positive_idx": dn_positive_idx,
        "cdn_num_group": num_group,
        "cdn_num_left": num_queries_left,
        "cdn_num_denoising": num_denoising,
        "cdn_num_right": num_queries_right,
    }

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
