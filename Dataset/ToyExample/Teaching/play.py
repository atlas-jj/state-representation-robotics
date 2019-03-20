import toy_model as model
from tkinter import *

if __name__ == '__main__':

    myM = model.PlayGroundBlocks()
    input("Press Enter to continue...")
    myM.refresh(400,700)
    # mainloop()  # execute this one will load the winForm UI.
    # gen = model.GenerateImages("blocks_c.npy", myM, "c_shape")
    # gen.save_samples("/home/chris/fc/bayesian_imitation/Experiments/Dataset", "Toy_Example")
