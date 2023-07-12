from .coco_det import Mscoco_det
from .custom import CustomDataset
from .mscoco import Mscoco
from .h36m import H36m
from .h36m_mpii import H36mMpii
from .mpii import Mpii

__all__ = ['CustomDataset', 'Mscoco', 'Mscoco_det', 'H36m', 'H36mMpii', 'Mpii']

datasets = {
    'mscoco': Mscoco,
    'mscoco_det': Mscoco_det,
    'h36m': H36m,
    'h36m_mpii': H36mMpii,
    'mpii': Mpii,
}

def build_datasets(dataset_cfg, preset_cfg):
    train_name = dataset_cfg['train']['type']
    val_name = dataset_cfg['val']['type']
    test_name = dataset_cfg['test']['type']
    # for phase in ['train', 'val', 'test']:
    #     dataset_cfg[phase]['preset'] = preset_cfg

    train_dataset = datasets[train_name](
        train=True,
        heatmap2coord=dataset_cfg['test']['heatmap2coord'],
        preset=preset_cfg,
        **dataset_cfg['train']
    )

    val_dataset = datasets[val_name](
        train=False,
        heatmap2coord=dataset_cfg['test']['heatmap2coord'],
        preset=preset_cfg,
        **dataset_cfg['val']
    )

    test_dataset = datasets[test_name](
        train=False,
        heatmap2coord=dataset_cfg['test']['heatmap2coord'],
        preset=preset_cfg,
         **dataset_cfg['test']
    )
    
    return train_dataset, val_dataset, test_dataset