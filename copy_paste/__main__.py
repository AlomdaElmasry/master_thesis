import skeltorch
import copy_paste.data
import copy_paste.runner

# Create Skeltorch object
skel = skeltorch.Skeltorch(data_type=copy_paste.data.CopyPasteData, runner_type=copy_paste.runner.CopyPasteRunner)

# Add the folder to store the data in test
skel.subparsers['test'].add_argument('--data-output', type=str, required=True, help='Where to store the output tests.')
skel.subparsers['test'].add_argument('--save-as-video', action='store_true',
                                     help='Whether or not to store the output as video.')

# Add another pipeline where only the alignment is tested
align_subparser = skel.create_subparser('test_alignment')
align_subparser.add_argument('--epoch', type=int, required=True, help='Epoch of the test.')
align_subparser.add_argument('--data-output', type=str, required=True, help='Where to store the output tests.')
align_subparser.add_argument('--save-as-video', action='store_true',
                             help='Whether or not to store the output as video.')
skel.create_command('test_alignment', skel.runner.test_alignment)

# Run execution using Skeltorch run() command
skel.run()
