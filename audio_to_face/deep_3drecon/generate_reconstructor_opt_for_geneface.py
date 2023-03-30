from options.test_options import TestOptions
import pickle as pkl

# run in the <audio_to_face> root dir!
opt = TestOptions().parse()  # get test options
opt.name='facerecon'
opt.epoch=20
opt.bfm_folder='deep_3drecon/BFM/'
opt.checkpoints_dir='deep_3drecon/checkpoints/'

with open("deep_3drecon/reconstructor_opt.pkl", 'wb') as f:
    pkl.dump(opt, f)
