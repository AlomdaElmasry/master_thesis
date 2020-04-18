import models.glunet
import os.path
import cv2


class ImagesAligner:
    _models_all = ['cpn_original', 'glunet']
    model = None

    def __init__(self, model_name='glunet'):
        assert model_name in self._models_all
        if model_name == 'glu-net':
            self._init_glunet()

    def _init_cpn_original(self, device):
        # Trained aligner should be in experiment named "". Checkpoint number is 1.
        exp_name = 'cpn_aligner_official'
        checkpoint_path = os.path.join(
            self.experiment.paths['checkpoints'].replace(self.experiment.experiment_name, exp_name), '1.checkpoint.pkl'
        )
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

        # Load model and state
        aligner_model = AlignerOriginal().to(device)
        aligner_model.load_state_dict(checkpoint_data['model'])

        # Freeze the state of the aligner
        for param in aligner_model.parameters():
            param.requires_grad = False

        # Return detached aligned with loaded weights
        return aligner_model

    def _init_glunet(self):
        checkpoint_path = None
        self.fmodel = models.glunet.GLU_Net(
            path_pre_trained_models='../weights/glunet/', model_type='DPED_CityScape_ADE',
            consensus_network=False, cyclic_consistency=True, iterative_refinement=True,
            apply_flipping_condition=False
        )

    def _init_cpn_weights(self):
        checkpoint_data = dict(torch.load('./weights/cpn.pth', map_location='cpu'))
        model_state = self.model.state_dict()
        for ck_item, k_data in checkpoint_data.items():
            if ck_item.replace('module.', '') in model_state:
                model_state[ck_item.replace('module.', '')].copy_(k_data)
        self.model.load_state_dict(model_state)

    def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
        """
        opencv remap : carefull here mapx and mapy contains the index of the future position for each pixel
        not the displacement !
        map_x contains the index of the future horizontal position of each pixel [i,j] while map_y contains the index of the
        future y position of each pixel [i,j]
        All are numpy arrays
        :param image: image to remap, HxWxC
        :param disp_x: displacement on the horizontal direction to apply to each pixel. must be float32. HxW
        :param disp_y: isplacement in the vertical direction to apply to each pixel. must be float32. HxW
        :return:
        remapped image. HxWxC
        """
        h_scale, w_scale = image.shape[:2]

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        map_x = (X + disp_x).astype(np.float32)
        map_y = (Y + disp_y).astype(np.float32)
        remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

        return remapped_image

    def align(self):
        pass
