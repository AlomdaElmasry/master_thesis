import skeltorch
from thesis.data import ThesisData
from .runner import ThesisAttentionRunner

# Create and run Skeltorch object
skeltorch.Skeltorch(ThesisData(), ThesisAttentionRunner()).run()
