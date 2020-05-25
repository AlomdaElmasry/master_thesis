import skeltorch
from thesis.data import ThesisData
from .runner import ThesisCorrelationRunner

# Create and run Skeltorch object
skeltorch.Skeltorch(ThesisData(), ThesisCorrelationRunner()).run()
