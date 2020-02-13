import skeltorch
import copy_paste.data
import copy_paste.runner

# Create Skeltorch object
skel = skeltorch.Skeltorch(data_type=copy_paste.data.CopyPasteData, runner_type=copy_paste.runner.CopyPasteRunner)

# Run execution using Skeltorch run() command
skel.run()
