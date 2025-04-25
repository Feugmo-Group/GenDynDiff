from gendyndiff.diffusion.data.batched_data import CustomCrystalDataset
from gendyndiff.common.data.collate import CustomCollate
from gendyndiff.common.data.chemgraph import ChemGraph

dump_file_path = "/home/agore/GenDynDiff/datasets/SrTiO3/dump.NPT"
datasets = CustomCrystalDataset.from_dump_file(dump_file_path)
print(datasets)


chemgraph_objects = []
for data_obj in datasets:
    chemgraph_obj = ChemGraph(pos=data_obj.pos)
    chemgraph_objects.append(chemgraph_obj)
    print(chemgraph_obj)

from torch_geometric.data import Batch
batched = Batch.from_data_list(chemgraph_objects)

print(batched)
print("Batched 'pos' shape:", batched.pos.shape)