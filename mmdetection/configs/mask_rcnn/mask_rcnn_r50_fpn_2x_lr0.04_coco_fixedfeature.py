_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x_lr0.04.py', '../_base_/default_runtime.py'
]

#checkpoint_config = dict(by_epoch=False, interval=100)
optimizer=dict(
    paramwise_cfg = dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0)
        }
    )
)
