from thesis.runner import ThesisRunner
import torch
import torch.nn.functional as F
import torch.utils.data
from .model_cpn import ThesisAligner
from .model_cpn_original import AlignerOriginal
import copy
import numpy as np
import os.path


class AlignerRunner(ThesisRunner):
    losses_items_ids = ['alignment']

    def init_model(self, device):
        self.model = AlignerOriginal().to(device)
        self._init_cpn_weights()

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-9
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.experiment.configuration.get('training', 'lr_scheduler_step_size'),
            gamma=self.experiment.configuration.get('training', 'lr_scheduler_gamma')
        )

    def _init_cpn_weights(self):
        checkpoint_data = dict(torch.load('./weights/cpn.pth', map_location='cpu'))
        model_state = self.model.state_dict()
        for ck_item, k_data in checkpoint_data.items():
            if ck_item.replace('module.', '') in model_state:
                model_state[ck_item.replace('module.', '')].copy_(k_data)
        self.model.load_state_dict(model_state)

    def train_step(self, it_data, device):
        self.save_states()
        exit()
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move Tensors to correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Propagate through the model
        (x_aligned, v_aligned, y_aligned), t, r_list = self.train_step_propagate(x, m, y)

        # Get visibility map of aligned frames and target frame
        visibility_maps = (1 - m[:, :, t].unsqueeze(2)) * v_aligned

        # Compute the loss
        loss = self._compute_loss(x[:, :, t], x_aligned, visibility_maps)

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        e_losses_items['alignment'].append(loss.item())

        # Return the loss
        return loss

    def train_step_propagate(self, x, m, y):
        # Set target and auxilliary frames indexes
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Return result from the model
        return self.model(x, m, y, t, r_list), t, r_list

    def test(self, epoch, device):
        # Set training parameters to the original test dataset
        samples_dataset = copy.deepcopy(self.experiment.data.datasets['train'])
        samples_dataset.frames_n = self.experiment.configuration.get('data', 'frames_n')

        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        subset_dataset = torch.utils.data.Subset(samples_dataset, self.experiment.data.test_frames_indexes)
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx = []
        x_aligned_tbx = []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, _ = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                (x_aligned, _, _), t, r_list = self.train_step_propagate(x, m, y)
            x_tbx.append(x.cpu().numpy())
            x_aligned_tbx.append(x_aligned.cpu().numpy())

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx)
        x_aligned_tbx = np.concatenate(x_aligned_tbx)

        # Save group image for each sample
        for b in range(len(self.experiment.data.test_frames_indexes)):
            default_list = [x_tbx[b, :, i] for i in range(x_tbx.shape[2])]
            aligned_list = [x_aligned_tbx[b, :, i] for i in range(x_aligned_tbx.shape[2] // 2)]
            aligned_list += [x_tbx[b, :, t]]
            aligned_list += [x_aligned_tbx[b, :, -i - 1] for i in range(x_aligned_tbx.shape[2] // 2)]
            self.experiment.tbx.add_images(
                'test_alignment_x/{}'.format(b), default_list, global_step=self.counters['epoch'], dataformats='CHW'
            )
            self.experiment.tbx.add_images(
                'test_alignment_x_aligned/{}'.format(b), aligned_list, global_step=self.counters['epoch'],
                dataformats='CHW'
            )

    def _compute_loss(self, x_t, x_aligned, v_map):
        alignment_input = x_aligned * v_map
        alignment_target = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1) * v_map
        return F.l1_loss(alignment_input, alignment_target, reduction='sum') / torch.sum(v_map)
