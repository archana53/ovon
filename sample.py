import json
import pickle as pkl

f = open("data/object_images_train.pickle", "rb")
map = pkl.load(f)
for thr in [0.025, 0.05, 0.1, 0.15, 0.20]:
    categories = []
    for obj in map.keys():
        if map[obj][0][1] > thr:
            categories.append(obj)
    out_file = open(f"data/obj/categories_{thr}.json", "w")
    json.dump(categories, out_file, indent=6)
    out_file.close()
