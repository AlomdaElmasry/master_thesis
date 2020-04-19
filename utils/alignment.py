import os.path
import cv2
import torch
import models.cpn_original
import numpy as np
import matplotlib.pyplot as plt


class AlignmentUtils:
    _models_names = ['cpn', 'glu-net']
    _models_init_handlers = {}
    _models_align_handlers = {}
    model_name = None
    model = None
    device = None

    def __init__(self, model_name, device):
        assert model_name in self._models_names
        self.model_name = model_name
        self.device = device
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

    def _init_glunet(self, device):
        import models.glunet
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'glunet')
        self.model = models.glunet.GLU_Net(
            path_pre_trained_models=checkpoint_path, model_type='DPED_CityScape_ADE',
            consensus_network=False, cyclic_consistency=True, iterative_refinement=True,
            apply_flipping_condition=False
        )

    def remap(self, image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR,
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
        source_images = x[:, :, t] * 255
        dest_images = x[:, :, t + 1] * 255
        print(source_images.size())
        print(dest_images.size())
        with torch.no_grad():
            estimated_flow = self.model.estimate_flow(source_images, dest_images, self.device, mode='channel_first')
            print(estimated_flow.size())
        warped_source_image = self.remap(
            source_images, estimated_flow.squeeze()[0].cpu().numpy(), estimated_flow.squeeze()[1].cpu().numpy()
        )

        fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
        axis1.imshow(source_images[0])
        axis1.set_title('Source image')
        axis2.imshow(dest_images[0])
        axis2.set_title('Target image')
        axis3.imshow(warped_source_image)
        axis3.set_title('Warped source image according to estimated flow by GLU-Net')
        fig.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Warped_source_image.png'),
                    bbox_inches='tight')
        plt.close(fig)
        exit()
