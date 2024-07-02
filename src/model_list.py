import timm
from pprint import pprint

# pprint(timm.list_models("*deit*"))
net = timm.create_model('deit3_medium_patch16_224')
print('Number of parameters(M):', sum(p.numel() for p in net.parameters()) / 1e6)
# print(net.named_modules)

# convnextv2_tiny 28 M
# regnetx_080 39.5 M
# deit3_medium_patch16_224 38.8 M
# resnet101 44.5 M