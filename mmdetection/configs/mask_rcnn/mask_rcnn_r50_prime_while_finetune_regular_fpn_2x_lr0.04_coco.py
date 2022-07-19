_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_prime.py',
    '../_base_/schedules/schedule_2x_lr0.04_with_prime.py', '../_base_/default_runtime.py'
]

