import skeltorch
from thesis.data import ThesisData
from .runner import AlignerRunner

# Create Skeltorch object and run it
skeltorch.Skeltorch(ThesisData(), AlignerRunner()).run()
