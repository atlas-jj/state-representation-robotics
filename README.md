# Implementation of State Representation Learning Methods in Robotics

Implementation of state representation methods (raw images) in robot hand eye coordination learning. States are raw images. Codes are based on pyTorch. I believe they are easy to read.

- Maintenance will be available when the author finds a full-time job. Sorry he has to raise two kids with very limited funding.
- More details are in this paper: [Evaluation of state representation methods in robot hand-eye coordination learning from demonstration][8] .

- <img src="https://github.com/atlas-jj/state-representation-robotics/blob/master/fig1.jpg?raw=true" width="400"/>

- If you find the codes are useful, please cite my [paper][8]. Citations are REALLY valuable, for a PhD student in a not so that famous research group.
- However, if you prefer not to cite due to various reasons (e.g., no enough space in your paper), I totally agree.


## Methods including: 
- [AE][1], auto encoder
- [VAE][2], variational auto encoder
- [beta-VAE][3]
- [Spatial Auto Encoder][4].


## Datasets
- toy example

<img src="https://github.com/atlas-jj/state-representation-robotics/blob/master/Dataset/ToyExample/Dataset/1/raw_104.jpg?raw=true" width="240"/>

- stack blocks

![Alt][5]

- fold clothes

![Alt][6]

## ToyExample/Teaching
- a toy robot model built on [TKinter][7]
- a human teaching interface
- support generating image sequence based on demonstrated trajectories.

## ToyExample/Control
- Robots: the robot model.
- main_env.py: environment to play the robot model.
- main_toy.py: main function to play with.

## Results/latent_space
- Results can be found under folder Results/each_task_specific_folder. E.g., an example of beta-VAE for the 'stack blocks' task.

![](https://github.com/atlas-jj/state-representation-robotics/blob/master/Results/Blocks/beta-VAE/BlocksV2_h1r1_dim50_0.05_172.8_1000_111678.82_4.34.jpg?raw=true)

## Results/TaskSpaceSampling
- how to sample: sampling every possible states in the toy_example task (2D). Each state correspondes to one image. Feed each image to the learned state representation method and read the value. Then visualize all sampled values in the 2D task space.
- beta-VAE: (one unit)

![](https://github.com/atlas-jj/state-representation-robotics/blob/master/Results/TaskSpaceSampling/Toy_example_z1_gap1_color.png?raw=true)

- SAE: (one unit)

![](https://github.com/atlas-jj/state-representation-robotics/blob/master/Results/TaskSpaceSampling/Toy_example_SAE_gap1_z2.png?raw=true)

## [Evaluation][8]
![](https://github.com/atlas-jj/state-representation-robotics/blob/master/fig2.jpg?raw=true)

[1]: https://en.wikipedia.org/wiki/Autoencoder
[2]: https://arxiv.org/abs/1312.6114
[3]: https://arxiv.org/abs/1804.03599
[4]: https://arxiv.org/abs/1509.06113
[5]: https://github.com/atlas-jj/state-representation-robotics/blob/master/Dataset/Blocks/final/raw_20.jpg?raw=true
[6]: https://github.com/atlas-jj/state-representation-robotics/blob/master/Dataset/Cloth/final/raw_20.jpg?raw=true
[7]: https://wiki.python.org/moin/TkInter
[8]: https://arxiv.org/abs/1903.00634
