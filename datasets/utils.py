def CreateDataset(args):
    if args.dataset.lower() == 'metabric':
        from .metabric_deephit import METABRICData
        return METABRICData(feature_file=args.feature_file, label_file=args.label_file, 
                            step=args.step, stratify=args.stratify, kfold=args.kfold, seed=args.seed, pad_left=args.pad_left, pad_right=args.pad_right)
    elif args.dataset.lower() == 'support':
        from .pycox_datasets import SupportDataset
        return SupportDataset(step=args.step, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize, pad_left=args.pad_left, pad_right=args.pad_right)
    elif args.dataset.lower() == 'gbsg':
        from .pycox_datasets import GBSGDataset
        return GBSGDataset(step=args.step, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize, pad_left=args.pad_left, pad_right=args.pad_right)
    elif args.dataset.lower() == 'flchain':
        from .pycox_datasets import FlchainDataset
        return FlchainDataset(step=args.step, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize, pad_left=args.pad_left, pad_right=args.pad_right)
    elif args.dataset.lower() == 'nwtco':
        from .pycox_datasets import NWTCODataSet
        return NWTCODataSet(step=args.step, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize, pad_left=args.pad_left, pad_right=args.pad_right)
    elif args.dataset.lower() == 'gbmlgg':
        from .tcga_gbmlgg import TcgaGbmLggData
        return TcgaGbmLggData(data_root=args.data_root, pickle_path=args.pickle_path, backbone=args.backbone, task=args.task,
                              n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed)
    elif args.dataset.lower() == 'external':
        from .external import RotterdamGBSGData
        return RotterdamGBSGData(step=args.step, seed=args.seed, pad_left=args.pad_left, pad_right=args.pad_right)