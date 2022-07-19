_base_ = [
    '../_base_/models/mask_rcnn_r50_adv_l2_eps0d1_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x_lr0.04.py', '../_base_/default_runtime.py'
]
