import thesis.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import torch
import utils.losses
import thesis_alignment.runner
import torch.nn.functional as F


class ThesisInpaintingRunner(thesis.runner.ThesisRunner):
    checkpoint_path = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test/checkpoints/58.checkpoint.pkl'
    model_vgg = None
    model_alignment = None
    utils_losses = None

    def init_model(self, device):
        torch.autograd.set_detect_anomaly(True)
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model_alignment = models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device)
        self.model = models.thesis_inpainting.ThesisInpaintingModel(self.model_vgg).to(device)
        self.load_alignment_state(self.checkpoint_path, device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.experiment.configuration.get('training', 'lr_scheduler_step_size'),
            gamma=self.experiment.configuration.get('training', 'lr_scheduler_gamma')
        )
        self.utils_losses = utils.losses.LossesUtils(device)
        self.losses_items_ids = ['recons_16', 'recons_64', 'recons_256']
        super().init_others(device)

    def load_alignment_state(self, checkpoint_path, device):
        with open(checkpoint_path, 'rb') as checkpoint_file:
            self.model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=device)['model'])

    def train_step(self, it_data, device):
        (x, m), y, info = it_data
        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

        # Compute t and r_list
        t, r_list = x.size(2) // 2, list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through alignment network
        with torch.no_grad():
            corr, xs, vs, ys, xs_aligned, _, vs_aligned, _, _, _, _, v_maps, _ = \
                thesis_alignment.runner.ThesisAlignmentRunner.train_step_propagate(
                    self.model_alignment, x, m, y, flow_gt, flows_use, t, r_list
                )

        # Propagate through inpainting network
        ys_hat, ys_hat_comp = ThesisInpaintingRunner.train_step_propagate(
            self.model, xs, vs, ys, xs_aligned, vs_aligned, v_maps, t, r_list
        )

        # Get both total loss and loss items
        loss, loss_items = ThesisInpaintingRunner.compute_loss(self.utils_losses, ys_hat, ys_hat_comp, ys, t, r_list)

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    def test(self, epoch, device):
        pass

    @staticmethod
    def train_step_propagate(model, xs, vs, ys, xs_aligned, vs_aligned, v_maps, t, r_list):
        y_hat_16, y_hat_comp_16, y_hat_64, y_hat_comp_64, y_hat_256, y_hat_comp_256 = model(
            [xs_item[:, :, t] for xs_item in xs], [vs_item[:, :, t] for vs_item in vs],
            [ys_item[:, :, t] for ys_item in ys], xs_aligned, vs_aligned, v_maps
        )
        return (y_hat_16, y_hat_64, y_hat_256), (y_hat_comp_16, y_hat_comp_64, y_hat_comp_256)

    @staticmethod
    def compute_loss(utils_losses, ys_hat, ys_hat_comp, ys, t, r_list):
        recons_16 = F.l1_loss(ys_hat[0], ys[0][:, :, t])
        recons_64 = F.l1_loss(ys_hat[1], ys[1][:, :, t])
        recons_256 = F.l1_loss(ys_hat[2], ys[2][:, :, t])
        loss = recons_16 + recons_64 + recons_256
        return loss, [recons_16, recons_64, recons_256]
