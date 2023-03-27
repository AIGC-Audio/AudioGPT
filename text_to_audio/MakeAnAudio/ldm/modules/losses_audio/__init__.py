from ldm.modules.losses_audio.vqperceptual import DummyLoss

# relative imports pain
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vggishish')
sys.path.append(path)
