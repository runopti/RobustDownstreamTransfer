_base_ = [
    '../_base_/models/upernet_r50_ant.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
optimizer=dict(
    paramwise_cfg = dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0)
        }
    )
)
#checkpoint_config = dict(by_epoch=False, interval=100)
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))
