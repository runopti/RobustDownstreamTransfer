import torch
import torch.nn as nn
import argparse
from collections import OrderedDict

def convert_advres(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.endswith('fc.weight') or k.endswith('fc.bias'):
            continue
        new_ckpt[k] = v
    return new_ckpt


def main():
   parser = argparse.ArgumentParser(
       description='Convert keys in official pretrained swin models to'
       'MMDetection style.')
   parser.add_argument('src', help='src detection model path')
   # The dst path must be a full path of the new checkpoint.
   parser.add_argument('dst', help='save path')
   args = parser.parse_args()

   checkpoint = torch.load(args.src, map_location='cpu')
   # module.model -> backbone
   
   if 'state_dict' in checkpoint:
       state_dict = checkpoint['state_dict']
   elif 'model' in checkpoint:
       state_dict = checkpoint['model']
   elif 'model_state_dict' in checkpoint:
       state_dict = checkpoint['model_state_dict']
   else:
       state_dict = checkpoint

   weight = convert_advres(state_dict)

   with open(args.dst, 'wb') as f:
       torch.save(weight, f)
    
if __name__ == '__main__':
   main()
