import skeltorch
from .data import EncoderDecoderData
from .runner import EncoderDecoderRunner

# Create Skeltorch object and run it
skeltorch.Skeltorch(EncoderDecoderData(), EncoderDecoderRunner()).run()