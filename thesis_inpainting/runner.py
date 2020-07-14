import thesis.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import models.thesis_inpainting
import torch
import utils.losses
import thesis_alignment.runner
import utils.flow
import utils.draws
import os.path
import matplotlib.pyplot as plt


class ThesisInpaintingRunner(thesis.runner.ThesisRunner):
    model_vgg = None
    model_alignment = None
    utils_losses = None

    def init_model(self, device):
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model_alignment = models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device)
        self.model = models.thesis_inpainting.ThesisInpaintingVisible().to(device)
        self.init_model_load_alignment_state(device)

    def init_model_load_alignment_state(self, device, experiment_name='align_double', epoch=66):
        experiment_path = os.path.join(os.path.dirname(self.experiment.paths['experiment']), experiment_name)
        checkpoint_path = os.path.join(experiment_path, 'checkpoints', '{}.checkpoint.pkl'.format(epoch))
        with open(checkpoint_path, 'rb') as checkpoint_file:
            self.model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=device)['model'])

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
        self.utils_losses = utils.losses.LossesUtils(self.model_vgg, device)
        self.losses_items_ids = ['loss_nh', 'loss_vh', 'loss_nvh', 'loss_perceptual']
        super().init_others(device)

    def train_step(self, it_data, device):
        (x, m), y, info = it_data
        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)
        t, r_list = self.get_indexes(x.size(2))

        # Compute t and r_list
        x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.train_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], y[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

        # Get both total loss and loss items
        loss, loss_items = ThesisInpaintingRunner.compute_loss(
            self.utils_losses, y[:, :, t], x_ref_aligned, (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    def test(self, epoch, device):
        self.model.eval()
        if epoch is not None:
            self.load_states(epoch, device)

        # Compute the losses on the test set
        self.test_losses(self.test_losses_handler, self.losses_items_ids, device)

        # Compute objective measures

        # Inpaint individual frames on the test set
        self.test_frames(self.test_frames_handler, 'validation', device)
        self.test_frames(self.test_frames_handler, 'test', device)

        # Inpaint test sequences every 10 epochs
        if epoch is not None or self.counters['epoch'] % 25 == 0:
            self.test_sequence(self.test_sequence_individual_handler, 'test_seq_individual', device)

    def test_losses_handler(self, x, m, y, flows_use, flow_gt, t, r_list):
        x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.infer_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], y[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )
        return ThesisInpaintingRunner.compute_loss(
            self.utils_losses, y[:, :, t], x_ref_aligned, (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )

    def test_frames_handler(self, x, m, y, t, r_list):
        return ThesisInpaintingRunner.infer_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], y[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

    def test_sequence_individual_handler(self, x, m, y):
        fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        y_inpainted = torch.zeros_like(x)
        for t in range(x.size(1)):
            self.logger.info('Step {}/{}'.format(t, x.size(1)))
            x_target, m_target, y_target, y_hat = x[:, t].unsqueeze(0), m[:, t].unsqueeze(0), y[:, t].unsqueeze(0), None
            t_candidates = ThesisInpaintingRunner.compute_priority_indexes(t, x.size(1), d_step=3, max_d=9E9)
            while (len(t_candidates) > 0 and torch.sum(m_target) * 100 / m_target.numel() > 1) or y_hat is None:
                r_index = [t_candidates.pop(0)]
                x_ref, m_ref = x[:, r_index].unsqueeze(0), m[:, r_index].unsqueeze(0)
                x_ref_aligned, _, v_map, y_hat, y_hat_comp = ThesisInpaintingRunner.infer_step_propagate(
                    self.model_alignment, self.model, x_target, m_target, y_target, x_ref, m_ref
                )
                m_target = m_target - v_map[:, :, 0]
                x_target = (1 - m_target) * y_hat_comp[:, :, 0] + m_target.repeat(1, 3, 1, 1) * fill_color
                y_target = (1 - m_target) * y_hat_comp[:, :, 0] + m_target.repeat(1, 3, 1, 1) * y_target
            y_inpainted[:, t] = x_target
        return y_inpainted

    @staticmethod
    def compute_priority_indexes(t, max_t, d_step, max_d):
        ref_candidates = list(range(max_t))
        ref_candidates.pop(t)
        ref_candidates_dist = list(map(lambda x: abs(x - t), ref_candidates))
        ref_candidates_sorted = [r[1] for r in sorted(zip(ref_candidates_dist, ref_candidates))]
        return list(
            filter(lambda x: abs(x - t) <= max_d and abs(x - t) % d_step == 0, ref_candidates_sorted)
        )

    @staticmethod
    def train_step_propagate(model_alignment, model, x_target, m_target, y_target, x_ref, m_ref):
        with torch.no_grad():
            x_ref_aligned, v_ref_aligned, v_map = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                model_alignment, x_target, m_target, x_ref, m_ref
            )
        y_hat, y_hat_comp = model(x_target, 1 - m_target, y_target, x_ref_aligned, v_ref_aligned, v_map)
        return x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp

    @staticmethod
    def infer_step_propagate(model_alignment, model, x_target, m_target, y_target, x_ref, m_ref):
        with torch.no_grad():
            x_ref_aligned, v_ref_aligned, v_map = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                model_alignment, x_target, m_target, x_ref, m_ref
            )
            y_hat, y_hat_comp = model(x_target, 1 - m_target, y_target, x_ref_aligned, v_ref_aligned, v_map)
        return x_ref_aligned, v_ref_aligned, v_map, y_hat, y_hat_comp

    @staticmethod
    def compute_loss(utils_losses, y_target, x_ref_aligned, v_target, y_hat, y_hat_comp, v_map):
        b, c, h, w = y_target.size()
        target_img = y_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        nh_mask = v_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        vh_mask = v_map
        nvh_mask = (1 - nh_mask) - vh_mask
        loss_nh = utils_losses.masked_l1(y_hat, target_img, nh_mask, reduction='sum', weight=0.25)
        loss_vh = utils_losses.masked_l1(y_hat, target_img, vh_mask, reduction='sum', weight=2)
        loss_vh_ref = utils_losses.masked_l1(y_hat, x_ref_aligned, vh_mask, reduction='sum', weight=0.10)
        loss_nvh = utils_losses.masked_l1(y_hat_comp, target_img, nvh_mask, reduction='sum', weight=0)
        loss_perceptual, *_ = utils_losses.perceptual(
            y_hat.transpose(1, 2).reshape(-1, c, h, w), target_img.transpose(1, 2).reshape(-1, c, h, w), weight=0.25
        )
        loss = loss_nh + loss_vh + loss_vh_ref + loss_nvh + loss_perceptual
        return loss, [loss_nh, loss_vh, loss_vh_ref, loss_perceptual]
