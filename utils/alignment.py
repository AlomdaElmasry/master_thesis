import os.path
import cv2
import torch
import models.cpn_original
import numpy as np


class AlignmentUtils:
    _models_names = ['cpn', 'glu-net']
    _models_init_handlers = {}
    _models_align_handlers = {}
    model_name = None
    model = None

    def __init__(self, model_name, device):
        assert model_name in self._models_names
        self.model_name = model_name
        self._models_init_handlers = {'cpn': self._init_cpn, 'glu-net': self._init_glunet}
        self._models_align_handlers = {'cpn': self._align_cpn, 'glu-net': self._align_glunet}
        self._models_init_handlers[model_name](device)

    def _init_cpn(self, device):
        self.model = models.cpn_original.CPNOriginalAligner().to(device)
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'cpn', 'cpn.pth')
        checkpoint_data = dict(torch.load(checkpoint_path, map_location=device))
        model_state = self.model.state_dict()
        for ck_item, k_data in checkpoint_data.items():
            if ck_item.replace('module.', '') in model_state:
                model_state[ck_item.replace('module.', '')].copy_(k_data)
        self.model.load_state_dict(model_state)
        for param in self.model.parameters():
            param.requires_grad = False

    def _init_glunet(self):
        import models.glunet
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'glunet')
        self.fmodel = models.glunet.GLU_Net(
            path_pre_trained_models=checkpoint_path, model_type='DPED_CityScape_ADE',
            consensus_network=False, cyclic_consistency=True, iterative_refinement=True,
            apply_flipping_condition=False
        )

    def remap_using_flow_fields(self, image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR,
                                border_mode=cv2.BORDER_CONSTANT):
        h_scale, w_scale = image.shape[:2]
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale), np.linspace(0, h_scale - 1, h_scale))
        map_x = (X + disp_x).astype(np.float32)
        map_y = (Y + disp_y).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    def align(self, x, m, y, t, r_list):
        return self._models_align_handlers[self.model_name](x, m, y, t, r_list)

    def _align_cpn(self, x, m, y, t, r_list):
        return self.model(x, m, y, t, r_list)

    def _align_glunet(self, x, m, y, t, r_list):
        print(self.model)
        exit()
