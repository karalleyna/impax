from pathlib import Path
from shutil import copy

import pandas as pd

df = pd.read_csv("all.csv")

shapenet_dir = Path("/mnt/d/ShapeNetCore.v2/ShapeNetCore.v2")
mesh_directory = Path("/mnt/d/ShapeNetCore.v2/data")

# strucute
# "[mesh_directory]/[splits]/[class names]/[ply files]

counter = 30
for file in shapenet_dir.glob("*/*/*/model_normalized.obj"):
    class_name = file.parent.parent.parent.name
    file_id = file.parent.parent.name
    filename = file.name
    
    print(class_name, file_id)
    split = str(df[(df.synsetId == int(class_name)) & (df.modelId == file_id)].split.values[0])
    out_dir = mesh_directory / split / class_name 
    
    out_dir.mkdir(exist_ok=True, parents=True)
    
    copy(str(file), str(out_dir / (file_id + ".obj")))
    if counter == 0:
        break
    counter -= 1