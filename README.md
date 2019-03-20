# Implementation of State Representation Learning Methods in Robotics
Implementation of state representation methods (raw images) in robot hand eye coordination learning. States are raw images.

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

[1]: https://en.wikipedia.org/wiki/Autoencoder
[2]: https://arxiv.org/abs/1312.6114
[3]: https://arxiv.org/abs/1804.03599
[4]: https://arxiv.org/abs/1509.06113
[5]: https://github.com/atlas-jj/state-representation-robotics/blob/master/Dataset/Blocks/final/raw_20.jpg?raw=true
[6]: https://github.com/atlas-jj/state-representation-robotics/blob/master/Dataset/Cloth/final/raw_20.jpg?raw=true
[7]: https://wiki.python.org/moin/TkInter

