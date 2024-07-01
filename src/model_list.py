import timm
from pprint import pprint

pprint(timm.list_models("*convnext*"))
# net = timm.create_model('convnext_nano')
# print('Number of parameters(M):', sum(p.numel() for p in net.parameters()) / 1e6)
# print(net.named_modules)