_base_ = [
    '../_base_/models/upernet_r50_prime_noJSD.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_lr0d005.py'
]
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))