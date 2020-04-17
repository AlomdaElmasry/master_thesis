import skeltorch
from .data import AlignerData
from .runner import AlignerRunner

# Create Skeltorch object and run it
skeltorch.Skeltorch(AlignerData(), AlignerRunner()).run()
