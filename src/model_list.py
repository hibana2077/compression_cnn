import timm
from pprint import pprint

pprint(timm.list_models("*dense*"))
# net = timm.create_model('convnextv2_huge')
# print('Number of parameters(M):', sum(p.numel() for p in net.parameters()) / 1e6)
# print(net)
# print(net.named_modules)
# model_list = timm.list_models("*convnext*")
# for model in model_list:
#     print(model)
# convnextv2_tiny 28 M
# regnetx_080 39.5 M
# deit3_medium_patch16_224 38.8 M
# resnet101 44.5 M