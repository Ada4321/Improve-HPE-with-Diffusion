from .datasets import DATASET_REGISTRY


def build_datasets(dataset_cfg):
    dataset_type = dataset_cfg["type"]

    train_dataset = DATASET_REGISTRY.get(dataset_type)(
        train=True,
        opt=dataset_cfg
    )
    val_dataset = DATASET_REGISTRY.get(dataset_type)(
        train=False,
        opt=dataset_cfg
    )
    
    return train_dataset, val_dataset

def build_datasets_debug(dataset_cfg, preset_cfg):
    train_name = dataset_cfg['train']['type']
    val_name = dataset_cfg['val']['type']
    test_name = dataset_cfg['test']['type']
    # for phase in ['train', 'val', 'test']:
    #     dataset_cfg[phase]['preset'] = preset_cfg

    train_dataset = DATASET_REGISTRY.get(train_name)(
        train=True,
        heatmap2coord=dataset_cfg['test']['heatmap2coord'],
        preset=preset_cfg,
        **dataset_cfg['train']
    )

    val_dataset = DATASET_REGISTRY.get(val_name)(
        train=False,
        heatmap2coord=dataset_cfg['test']['heatmap2coord'],
        preset=preset_cfg,
        **dataset_cfg['val']
    )

    test_dataset = DATASET_REGISTRY.get(test_name)(
        train=False,
        heatmap2coord=dataset_cfg['test']['heatmap2coord'],
        preset=preset_cfg,
         **dataset_cfg['test']
    )
    
    return train_dataset, val_dataset, test_dataset