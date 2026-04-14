import torch, os

# tile_dir = "/data-hdd0/Ethan_Baird/Dec25_xenium/processed_tiles/train_tiles/processed"

# bad = 0
# total = 0

# for f in os.listdir(tile_dir):
#     path = os.path.join(tile_dir, f)

#     # skip directories
#     if not os.path.isfile(path):
#         continue

#     d = torch.load(path, weights_only=False)
#     total += 1

#     if not hasattr(d["tx"], "x") or not hasattr(d["nc"], "x"):
#         bad += 1

# print("bad:", bad, "total:", total)


tile_dir = "/data-hdd0/Ethan_Baird/Dec25_xenium/processed_tiles/train_tiles/processed"

# list files in the folder
files = os.listdir(tile_dir)
print("First 10 files:", files[:10])

# pick the first file to inspect
tile_path = os.path.join(tile_dir, files[0])
print("Inspecting:", tile_path)

d = torch.load(tile_path, weights_only=False)

print("Keys in data object:", d.keys())
print("tx:", d["tx"])
print("nc:", d["nc"])