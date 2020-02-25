import skeltorch
from copy_paste.data import CopyPasteData
from copy_paste.runner import CopyPasteRunner

# Create Skeltorch object
runner = CopyPasteRunner()
skel = skeltorch.Skeltorch(CopyPasteData(), runner)

# Add the folder to store the data in test
test_parser = skel.get_subparser('test')
test_parser.add_argument('--data-output', type=str, required=True, help='Where to store the output tests.')
test_parser.add_argument('--save-as-video', action='store_true', help='Whether or not to store the output as video.')

# Add another pipeline where only the alignment is tested
align_subparser = skel.create_subparser('test_alignment')
align_subparser.add_argument('--epoch', type=int, required=True, help='Epoch of the test.')
align_subparser.add_argument('--data-output', type=str, required=True, help='Where to store the output tests.')
align_subparser.add_argument('--save-as-video', action='store_true', help='Whether store the output as video.')
skel.create_command('test_alignment', runner.test_alignment, ['epoch', 'data_output', 'save_as_video'])

# Run execution using Skeltorch run() command
skel.run()
