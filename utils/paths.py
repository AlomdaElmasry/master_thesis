import os.path
import glob


class DatasetPaths:
    dataset_paths = {'davis-2017': 'DAVIS-2017', 'got-10k': 'GOT10k', 'youtube-vos': 'YouTubeVOS'}

    @staticmethod
    def get_items(dataset_name, data_folder, split):
        dataset_folder = os.path.join(data_folder, DatasetPaths.dataset_paths[dataset_name])
        if dataset_name == 'davis-2017':
            return DatasetPaths.get_davis(dataset_folder, split)
        elif dataset_name == 'got-10k':
            return DatasetPaths.get_got10k(dataset_folder, split)
        elif dataset_name == 'youtube-vos':
            return DatasetPaths.get_youtube_vos(dataset_folder, split)

    @staticmethod
    def get_davis(dataset_folder, split):
        split_filename = 'train.txt' if split == 'train' else 'val.txt'
        with open(os.path.join(dataset_folder, 'ImageSets', '2017', split_filename)) as items_file:
            items_names = items_file.read().splitlines()
        items_gts_paths = [sorted(glob.glob(
            os.path.join(dataset_folder, 'JPEGImages', '480p', item_name, '*.jpg'))) for item_name in items_names
        ]
        items_masks_paths = [sorted(glob.glob(
            os.path.join(dataset_folder, 'Annotations', '480p', item_name, '*.png'))) for item_name in items_names
        ]
        return items_names, items_gts_paths, items_masks_paths


    @staticmethod
    def get_got10k(dataset_folder, split):
        split_folder = 'train' if split == 'train' else 'val' if split == 'validation' else 'test'
        with open(os.path.join(dataset_folder, split_folder, 'list.txt')) as items_file:
            items_names = items_file.read().splitlines()
        items_gts_paths = [sorted(glob.glob(
            os.path.join(dataset_folder, split_folder, item_name, '*.jpg'))) for item_name in items_names
        ]
        items_masks_paths = None
        return items_names, items_gts_paths, items_masks_paths


    @staticmethod
    def get_youtube_vos(dataset_folder, split):
        split_folder = 'train' if split == 'train' else 'valid' if split == 'validation' else 'test'
        items_names = os.listdir(os.path.join(dataset_folder, split_folder, 'JPEGImages'))
        items_gts_paths = [
            sorted(glob.glob(os.path.join(dataset_folder, split_folder, 'JPEGImages', sequence_name, '*.jpg')))
            for sequence_name in items_names
        ]
        items_masks_paths = [
            sorted(glob.glob(os.path.join(dataset_folder, split_folder, 'Annotations', sequence_name, '*.png')))
            for sequence_name in items_names
        ]
        return items_names, items_gts_paths, items_masks_paths
