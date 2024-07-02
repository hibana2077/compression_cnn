import timm
from tqdm import tqdm
from pprint import pprint

model_list = timm.list_models("")
search_value = 38176484/1e6
quilified_models = []
for model in tqdm(model_list):
    net = timm.create_model(model)
    if -10 <= (sum(p.numel() for p in net.parameters()) / 1e6 - search_value) <= 10:
        quilified_models.append(model)

pprint(quilified_models)
# net = timm.create_model('deit3_medium_patch16_224')
# print('Number of parameters(M):', sum(p.numel() for p in net.parameters()) / 1e6)
# print(net.named_modules)

# convnextv2_tiny 28 M
# regnetx_080 39.5 M
# deit3_medium_patch16_224 38.8 M
# resnet101 44.5 M