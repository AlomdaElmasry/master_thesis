import os.path
import glob


class DatasetPaths:
    dataset_paths = {'davis-2017': 'DAVIS-2017', 'got-10k': 'GOT10k', 'youtube-vos': 'YouTubeVOS'}

    @staticmethod
    def get_items(dataset_name, data_folder, split, return_gts=True, return_masks=True):
        assert return_gts or return_masks
        dataset_folder = os.path.join(data_folder, DatasetPaths.dataset_paths[dataset_name])
        if dataset_name == 'davis-2017':
            return DatasetPaths.get_davis(dataset_folder, split, return_gts, return_masks)
        elif dataset_name == 'got-10k':
            return DatasetPaths.get_got10k(dataset_folder, split, return_gts, return_masks)
        elif dataset_name == 'youtube-vos':
            return DatasetPaths.get_youtube_vos(dataset_folder, split, return_gts, return_masks)

    @staticmethod
    def get_davis(dataset_folder, split, return_gts, return_masks):
        split_filename = 'train.txt' if split == 'train' else 'val.txt'
        items_file = open(os.path.join(dataset_folder, 'ImageSets', '2017', split_filename))
        items_meta = {}
        for item_name in sorted(items_file.read().splitlines()):
            item_gts_paths = sorted(
                glob.glob(os.path.join(dataset_folder, 'JPEGImages', '480p', item_name, '*.jpg'))
            )
            item_masks_path = sorted(
                glob.glob(os.path.join(dataset_folder, 'Annotations', '480p', item_name, '*.png'))
            )
            items_meta[item_name] = (item_gts_paths, item_masks_path)
        return items_meta

    @staticmethod
    def get_got10k(dataset_folder, split, return_gts, return_masks):
        split_folder = 'train' if split == 'train' else 'val' if split == 'validation' else 'test'
        items_file = open(os.path.join(dataset_folder, split_folder, 'list.txt'))
        items_meta = {}
        for item_name in sorted(items_file.read().splitlines()):
            if os.path.exists(os.path.join(dataset_folder, split_folder, item_name)):
                item_gts_paths = sorted(glob.glob(os.path.join(dataset_folder, split_folder, item_name, '*.jpg')))
                if len(item_gts_paths) > 0:
                    items_meta[item_name] = (item_gts_paths, None)
        return items_meta

    @staticmethod
    def get_youtube_vos(dataset_folder, split, return_gts, return_masks):
        split_folder = 'train' if split == 'train' else 'valid' if split == 'validation' else 'test'
        type_folder = 'JPEGImages' if return_gts else 'Annotations'
        items_meta = {}
        for item_name in sorted(os.listdir(os.path.join(dataset_folder, split_folder, type_folder))):
            item_gts_paths = sorted(
                glob.glob(os.path.join(dataset_folder, split_folder, 'JPEGImages', item_name, '*.jpg'))
            ) if return_gts else None
            item_masks_paths = sorted(
                glob.glob(os.path.join(dataset_folder, split_folder, 'Annotations', item_name, '*.png'))
            ) if return_masks else None
            items_meta[item_name] = (item_gts_paths, item_masks_paths)
        return items_meta
