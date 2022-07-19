_base_ = [
    '../_base_/models/mask_rcnn_r50_prime_noJSD_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x_lr0.01.py', '../_base_/default_runtime.py'
]
