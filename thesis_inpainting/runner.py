import thesis.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import models.thesis_inpainting
import torch
import utils.losses
import thesis_alignment.runner
import torch.nn.functional as F
import utils.flow
import numpy as np
import utils.draws
import matplotlib.pyplot as plt


class ThesisInpaintingRunner(thesis.runner.ThesisRunner):
    checkpoint_path_cuda = '/home/ubuntu/ebs/master_thesis/experiments/align_v3_1/checkpoints/45.checkpoint.pkl'
    checkpoint_path_cpu = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test/checkpoints/45.checkpoint.pkl'
    model_vgg = None
    model_alignment = None
    utils_losses = None

    def init_model(self, device):
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model_alignment = models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device)
        self.model = models.thesis_inpainting.ThesisInpaintingVisible().to(device)
        self.load_alignment_state(device)

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

    def load_alignment_state(self, device):
        checkpoint_path = self.checkpoint_path_cuda if device == 'cuda' else self.checkpoint_path_cpu
        with open(checkpoint_path, 'rb') as checkpoint_file:
            self.model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=device)['model'])

    def train_step(self, it_data, device):
        (x, m), y, info = it_data
        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)
        t, r_list = self.get_indexes(x.size(2))

        # Compute t and r_list
        y_hat, y_hat_comp, v_map, *_ = ThesisInpaintingRunner.train_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], y[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

        # Get both total loss and loss items
        loss, loss_items = ThesisInpaintingRunner.compute_loss(
            self.utils_losses, y[:, :, t], (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
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
        if epoch is not None or self.counters['epoch'] % 10 == 0:
            self.test_sequence(self.test_sequence_individual_handler, 'test_seq_individual', device)
        self.test_frames(self.test_frames_handler, device)

    def test_frames_handler(self, x, m, y, t, r_list):
        return ThesisInpaintingRunner.infer_step_propagate(
            self.model_alignment, self.model, x[:, :, t], m[:, :, t], y[:, :, t], x[:, :, r_list], m[:, :, r_list]
        )

    def test_sequence_individual_handler(self, x, m, y):
        fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(x.device)
        y_inpainted = torch.zeros_like(x)
        for t in range(x.size(1)):
            self.logger.info('Step {}/{}'.format(t, x.size(1)))
            x_target, m_target, y_target = x[:, t].unsqueeze(0), m[:, t].unsqueeze(0), y[:, t].unsqueeze(0)
            t_candidates = ThesisInpaintingRunner.compute_priority_indexes(t, x.size(1), d_step=3, max_d=9E9)
            while len(t_candidates) > 0 and torch.sum(m_target) * 100 / m_target.numel() > 1:
                r_index = [t_candidates.pop(0)]
                x_ref, m_ref = x[:, r_index].unsqueeze(0), m[:, r_index].unsqueeze(0)
                y_hat, y_hat_comp, v_map, x_ref_aligned, _ = ThesisInpaintingRunner.infer_step_propagate(
                    self.model_alignment, self.model, x_target, m_target, y_target, x_ref, m_ref
                )
                m_target = m_target - v_map[:, :, 0]
                x_target = (1 - m_target) * y_hat_comp[:, :, 0] + m_target.repeat(1, 3, 1, 1) * fill_color
                y_target = (1 - m_target) * y_hat_comp[:, :, 0] + m_target.repeat(1, 3, 1, 1) * y_target
            y_inpainted[:, t] = y_hat[0, :, 0]
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
            x_ref_aligned, v_ref_aligned = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                model_alignment, x_target, m_target, x_ref, m_ref
            )
        y_hat, y_hat_comp, v_map = model(x_target, 1 - m_target, y_target, x_ref_aligned, v_ref_aligned)
        return y_hat, y_hat_comp, v_map, x_ref_aligned, v_ref_aligned

    @staticmethod
    def infer_step_propagate(model_alignment, model, x_target, m_target, y_target, x_ref, m_ref):
        with torch.no_grad():
            x_ref_aligned, v_ref_aligned = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                model_alignment, x_target, m_target, x_ref, m_ref
            )
            y_hat, y_hat_comp, v_map = model(x_target, 1 - m_target, y_target, x_ref_aligned, v_ref_aligned)
        return y_hat, y_hat_comp, v_map, x_ref_aligned, v_ref_aligned

    @staticmethod
    def compute_loss(utils_losses, y_target, v_target, y_hat, y_hat_comp, v_map):
        b, c, h, w = y_target.size()
        target_img = y_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        nh_mask = v_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        vh_mask = v_map
        nvh_mask = (1 - nh_mask) - vh_mask
        loss_nh = utils_losses.masked_l1(y_hat_comp, target_img, nh_mask, reduction='sum', weight=0)
        loss_vh = utils_losses.masked_l1(y_hat_comp, target_img, vh_mask, reduction='sum', weight=2)
        loss_nvh = utils_losses.masked_l1(y_hat_comp, target_img, nvh_mask, reduction='sum', weight=0.5)
        loss_perceptual, *_ = utils_losses.perceptual(
            y_hat_comp.transpose(1, 2).reshape(-1, c, h, w), target_img.transpose(1, 2).reshape(-1, c, h, w), weight=1
        )
        loss = loss_nh + loss_vh + loss_nvh + loss_perceptual
        return loss, [loss_nh, loss_vh, loss_nvh, loss_perceptual]
