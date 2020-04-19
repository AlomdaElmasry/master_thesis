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

    def align(self, x, m, y, t, r_list):
        return self._models_align_handlers[self.model_name](x, m, y, t, r_list)

    def _align_cpn(self, x, m, y, t, r_list):
        return self.model(x, m, y, t, r_list)

    def _align_glunet(self, x, m, y, t, r_list):
        # Get input dimensions
        b, c, f, h, w = x.size()

        # Define target and aux frames of shape (b, 3, f - 1, h, w)
        target_frame = (x[:, :, t] * 255).byte().unsqueeze(2).repeat(1, 1, f - 1, 1, 1)
        aux_frames = (x[:, :, r_list] * 255).byte()

        # Expand to batch dimension
        target_frame = target_frame.transpose(1, 2).reshape(-1, c, h, w)
        aux_frames = aux_frames.transpose(1, 2).reshape(-1, c, h, w)

        # Estimate the flows
        with torch.no_grad():
            estimated_flow = self.model.estimate_flow(aux_frames, target_frame, self.device, mode='channel_first')

        # Get aligned images
        x_aligned = self._align_glunet_transform((aux_frames / 255).float(), estimated_flow)
        print(x_aligned.size())

        # Testing with torch
        # warped_source_image = self._align_glunet_transform(x[:, :, t], estimated_flow)

        # fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
        # axis1.imshow(aux_frames[0].permute(1, 2, 0).cpu().numpy())
        # axis1.set_title('Source image')
        # axis2.imshow(target_frame[0].permute(1, 2, 0).cpu().numpy())
        # axis2.set_title('Target image')
        # axis3.imshow(warped_source_image[0].permute(1, 2, 0).cpu())
        # axis3.set_title('Warped source image according to estimated flow by GLU-Net')
        # fig.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Warped_source_image.png'),
        #             bbox_inches='tight')
        # plt.close(fig)

        exit()

    def _align_glunet_transform(self, image, estimated_flow):
        # Image is FloatTensor of size (16, 3, 256, 256)
        # EstimatedFlow is FloatTensor of size (16, 2, 256, 256)
        # IdentityGrid is FloatTensor of size (16, 256, 256, 2)
        identity_theta = self.movement_simulator.identity_theta(image.size(2), image.size(3)) \
            .unsqueeze(0).expand(image.size(0), 2, 3).to(self.device)
        identity_grid = torch.nn.functional.affine_grid(identity_theta, image.size())

        # Add normalized x and y displacement
        identity_grid[:, :, :, 0] += estimated_flow[:, 0] / 256
        identity_grid[:, :, :, 1] += estimated_flow[:, 1] / 256

        # Apply transformation to the image
        return torch.nn.functional.grid_sample(image, identity_grid)
