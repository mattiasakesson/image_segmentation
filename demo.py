from u_net import Unet_model
import pickle

validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )[:,:,species]

model = Unet_model()

