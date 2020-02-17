import skeltorch
import copy_paste.data
import copy_paste.runner

# Create Skeltorch object
skel = skeltorch.Skeltorch(data_type=copy_paste.data.CopyPasteData, runner_type=copy_paste.runner.CopyPasteRunner)

# Add Alignment command
align_subparser = skel.create_subparser('test_alignment')
align_subparser.add_argument('--epoch', type=int, required=True, help='Epoch of the test.')
skel.create_command('test_alignment', skel.runner.test_alignment)

# Run execution using Skeltorch run() command
skel.run()
