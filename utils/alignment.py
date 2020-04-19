import os.path
import cv2
import torch
import models.cpn_original
import numpy as np
import matplotlib.pyplot as plt
import utils.movement


class AlignmentUtils:
    _models_names = ['cpn', 'glu-net']
    _models_init_handlers = {}
    _models_align_handlers = {}
    model_name = None
    model = None
    device = None
    movement_simulator = None

    def __init__(self, model_name, device):
        assert model_name in self._models_names
        self.model_name = model_name
        self.device = device
        self.movement_simulator = utils.movement.MovementSimulator()
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

        # h_scale = 256, w_scale = 256
        h_scale, w_scale = image.shape[:2]

        # Linspace between 0 and 255, 256 points
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale), np.linspace(0, h_scale - 1, h_scale))
        map_x = (X + disp_x).astype(np.float32)
        map_y = (Y + disp_y).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    def map_torch(self, image, estimated_flow):
        # Image is FloatTensor of size (16, 3, 256, 256)
        # EstimatedFlow is FloatTensor of size (16, 2, 256, 256)
        identity_theta = self.movement_simulator.identity_theta(image.size(2), image.size(3))
        print(identity_theta)
        # x_flow = torch.linspace()
        print(image.size())
        print(estimated_flow.size())
        exit()
        pass

    def align(self, x, m, y, t, r_list):
        return self._models_align_handlers[self.model_name](x, m, y, t, r_list)

    def _align_cpn(self, x, m, y, t, r_list):
        return self.model(x, m, y, t, r_list)

    def _align_glunet(self, x, m, y, t, r_list):
        source_images = (x[:, :, t] * 255).byte()
        dest_images = (x[:, :, t + 1] * 255).byte()
        with torch.no_grad():
            estimated_flow = self.model.estimate_flow(source_images, dest_images, self.device, mode='channel_first')
        warped_source_image = self.map_torch(source_images, estimated_flow)

        # warped_source_image = self.remap(
        #     source_images[0].permute(1, 2, 0).cpu().numpy(), estimated_flow[0, 0].cpu().numpy(), estimated_flow[0, 1].cpu().numpy()
        # )

        fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
        axis1.imshow(source_images[0].permute(1, 2, 0).cpu().numpy())
        axis1.set_title('Source image')
        axis2.imshow(dest_images[0].permute(1, 2, 0).cpu().numpy())
        axis2.set_title('Target image')
        axis3.imshow(warped_source_image)
        axis3.set_title('Warped source image according to estimated flow by GLU-Net')
        fig.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Warped_source_image.png'),
                    bbox_inches='tight')
        plt.close(fig)
        exit()
