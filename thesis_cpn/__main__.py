import skeltorch
from thesis.data import ThesisData
from .runner import CopyPasteRunner

# Create Skeltorch object
runner = CopyPasteRunner()
skel = skeltorch.Skeltorch(ThesisData(), runner)

# Add the folder to store the data in test
test_parser = skel.get_parser('test')
test_parser.add_argument('--save-as-video', action='store_true', help='Whether or not to store the output as video.')
skel.create_command(test_parser, runner.test, ['epoch', 'save_as_video', 'device'])

# Add a pipeline to test the alignment
align_subparser = skel.create_parser('test_alignment')
align_subparser.add_argument('--epoch', type=int, required=True, help='Epoch of the test.')
align_subparser.add_argument('--save-as-video', action='store_true', help='Whether store the output as video.')
align_subparser.add_argument('--device', default='cpu', help='PyTorch-friendly device name.')
skel.create_command(align_subparser, runner.test_alignment, ['epoch', 'save_as_video', 'device'])

# Add a pipeline to test the inpainting
inpainting_subparser = skel.create_parser('test_inpainting')
inpainting_subparser.add_argument('--epoch', type=int, required=True, help='Epoch of the test.')
inpainting_subparser.add_argument('--save-as-video', action='store_true', help='Whether store the output as video.')
inpainting_subparser.add_argument('--device', default='cpu', help='PyTorch-friendly device name.')
skel.create_command(inpainting_subparser, runner.test_inpainting, ['epoch', 'save_as_video', 'device'])

# Run execution using Skeltorch run() command
skel.run()
