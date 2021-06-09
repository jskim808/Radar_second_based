import time
from enum import Enum
from functools import reduce
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.models.voxelnet import register_voxelnet, VoxelNet
from second.pytorch.models import rpn


class SmallObjectHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        final_num_filters = 64
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict


class DefaultHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        final_num_filters = num_filters
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict

@register_voxelnet
class VoxelNetNuscenesMultiHead(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)
        self.small_classes = ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
        self.large_classes = ["car", "truck", "trailer", "bus", "construction_vehicle"]
        small_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        large_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])
        self.small_head = SmallObjectHead(
            num_filters=self.rpn._num_filters[0],
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )
        self.large_head = DefaultHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )

    def network_forward(self, voxels, num_points, coors, batch_size):
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)
        self.end_timer("middle forward")
        self.start_timer("rpn forward")
        rpn_out = self.rpn(spatial_features)
        r1 = rpn_out["stage0"]
        _, _, H, W = r1.shape
        cropsize40x40 = np.round(H * 0.1).astype(np.int64)
        r1 = r1[:, :, cropsize40x40:-cropsize40x40, cropsize40x40:-cropsize40x40]
        small = self.small_head(r1)
        large = self.large_head(rpn_out["out"])
        self.end_timer("rpn forward")
        # concated preds MUST match order in class_settings in config.
        res = {
            "box_preds": torch.cat([large["box_preds"], small["box_preds"]], dim=1),
            "cls_preds": torch.cat([large["cls_preds"], small["cls_preds"]], dim=1),
        }
        if self._use_direction_classifier:
            res["dir_cls_preds"] = torch.cat([large["dir_cls_preds"], small["dir_cls_preds"]], dim=1)
        return res


@register_voxelnet
class MyVoxelNetNuscenesMultiHead(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)

        # self.small_classes = ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
        # self.large_classes = ["car", "truck", "trailer", "bus", "construction_vehicle"]

        self.large_class_car = ["car"]
        self.large_class_bus_trailer = ["bus", "trailer"]
        self.large_class_truck_cons_veh = ["truck", "construction_vehicle"]

        self.small_class_pedes = ["pedestrian"]
        self.small_class_tr_cone = ["traffic_cone"]
        self.small_class_barrier = ["barrier"]
        self.small_class_bicycle_moto = ["bicycle", "motorcycle"]


        # small_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        # large_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])

        car_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_class_car])
        bus_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_class_bus_trailer])
        truck_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_class_truck_cons_veh])

        pedestrian_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_class_pedes])
        traffic_cone_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_class_tr_cone])
        barrier_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_class_barrier])
        bicycle_anchor = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_class_bicycle_moto])

        self.car_head = DefaultHead(
            # num_filters=np.sum(self.rpn._num_upsample_filters),
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=car_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )

        self.bus_head = DefaultHead(
            # num_filters=np.sum(self.rpn._num_upsample_filters),
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=bus_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )


        self.truck_head = DefaultHead(
            # num_filters=np.sum(self.rpn._num_upsample_filters),
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=truck_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )


        self.pedestrian_head = DefaultHead(
            # num_filters=self.rpn._num_filters[0],
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=pedestrian_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )


        self.traffic_cone_head = DefaultHead(
            # num_filters=self.rpn._num_filters[0],
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=traffic_cone_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )

        self.barrier_head = DefaultHead(
            # num_filters=self.rpn._num_filters[0],
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=barrier_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )

        self.bicycle_head = DefaultHead(
            # num_filters=self.rpn._num_filters[0],
            num_filters=384,
            num_class=self._num_class,
            num_anchor_per_loc=bicycle_anchor,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size, 
            num_direction_bins=self._num_direction_bins,
        )


    def network_forward(self, voxels, num_points, coors, batch_size):
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)
        self.end_timer("middle forward")

        self.start_timer("rpn forward")

        rpn_out = self.rpn(spatial_features)

        # r1 = rpn_out["stage0"]
        # _, _, H, W = r1.shape
        # cropsize40x40 = np.round(H * 0.1).astype(np.int64)
        # r1 = r1[:, :, cropsize40x40:-cropsize40x40, cropsize40x40:-cropsize40x40]
        # small = self.small_head(r1)
        # large = self.large_head(rpn_out["out"])

        rpn_features = rpn_out["out"]
        #lidar_features = rpn_out["out"]

        car = self.car_head(rpn_features)
        bus = self.bus_head(rpn_features)
        truck = self.truck_head(rpn_features)

        pedestrian = self.pedestrian_head(rpn_features)
        traffic_cone = self.traffic_cone_head(rpn_features)
        barrier = self.barrier_head(rpn_features)
        bicycle = self.bicycle_head(rpn_features)

        self.end_timer("rpn forward")
        
        # concated preds MUST match order in class_settings in config.
        
        # res = {
        #     "box_preds": torch.cat([large["box_preds"], small["box_preds"]], dim=1),
        #     "cls_preds": torch.cat([large["cls_preds"], small["cls_preds"]], dim=1),
        # }

        res = {
            "box_preds": torch.cat([car["box_preds"], bus["box_preds"], truck["box_preds"],
                                    pedestrian["box_preds"], traffic_cone["box_preds"], barrier["box_preds"], 
                                    bicycle["box_preds"]], dim=1),

            "cls_preds": torch.cat([car["cls_preds"], bus["cls_preds"], truck["cls_preds"],
                                    pedestrian["cls_preds"], traffic_cone["cls_preds"], barrier["cls_preds"], 
                                    bicycle["cls_preds"]], dim=1),
        }

        # if self._use_direction_classifier:
            # res["dir_cls_preds"] = torch.cat([large["dir_cls_preds"], small["dir_cls_preds"]], dim=1)

        res["dir_cls_preds"] = torch.cat([car["dir_cls_preds"], bus["dir_cls_preds"], 
                                truck["dir_cls_preds"], pedestrian["dir_cls_preds"], traffic_cone["dir_cls_preds"], 
                                barrier["dir_cls_preds"], bicycle["dir_cls_preds"]], dim=1)

        return res
