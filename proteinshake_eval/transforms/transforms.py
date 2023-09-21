from .graph import GraphTrainTransform, GraphPairTrainTransform, ProteinEdgeTypeTransform
from .point2 import PointTrainTransform, PointPairTrainTransform
from .voxel import VoxelTrainTransform, VoxelPairTrainTransform
from .utils import PPIDataset
from proteinshake.transforms import Compose
from .graph import GraphPretrainTransform, MaskNode
from .point2 import PointPretrainTransform, MaskPoint
from .voxel import VoxelPretrainTransform
from .graph import GraphLineTransform


def get_transformed_dataset(cfg, dataset, task, y_transform=None):
    if 'pair' in task.task_type[0]:
        if cfg.name == 'graph':
            data_transform = GraphPairTrainTransform()
            dataset = dataset.to_graph(eps=cfg.graph_eps).pyg()
        elif cfg.name == 'point_cloud':
            data_transform = PointPairTrainTransform()
            dataset = dataset.to_point().torch()
        elif cfg.name == 'voxel':
            data_transform = VoxelPairTrainTransform()
            dataset = dataset.to_voxel(
                gridsize=cfg.gridsize, voxelsize=cfg.voxelsize).torch()
        else:
            raise ValueError("Not implemented!")

        train_dset = PPIDataset(dataset, task, 'train', transform=data_transform, y_transform=y_transform)
        val_dset = PPIDataset(dataset, task, 'val', transform=data_transform, y_transform=y_transform)
        test_dset = PPIDataset(dataset, task, 'test', transform=data_transform, y_transform=y_transform)
        return train_dset, val_dset, test_dset

    if cfg.name == 'graph':
        data_transform = GraphTrainTransform(task, y_transform)
        return dataset.to_graph(eps=cfg.graph_eps).pyg(transform=data_transform)
    elif cfg.name == 'point_cloud':
        data_transform = PointTrainTransform(task, y_transform)
        return dataset.to_point().torch(transform=data_transform)
    elif cfg.name == 'voxel':
        data_transform = VoxelTrainTransform(task, y_transform)
        return dataset.to_voxel(
            gridsize=cfg.gridsize, voxelsize=cfg.voxelsize).torch(transform=data_transform)
    else:
        raise ValueError("Not implemented!")

def get_pretrain_dataset(cfg, dataset):
    if cfg.task.name in ['alphafold', 'rcsb']: 
        if cfg.training.strategy == 'mask':
            data_transform = Compose([
                GraphPretrainTransform(), ProteinEdgeTypeTransform(), MaskNode(20, mask_rate=cfg.representation.mask_rate)
            ])
        if cfg.training.strategy == 'motif':
            data_transform = Compose([GraphPretrainTransform(), ProteinEdgeTypeTransform(), GraphLineTransform()])

        return dataset.to_graph(eps=cfg.representation.graph_eps).pyg(transform=data_transform)
    else:
        raise ValueError("Dataset Not Implemented.")
