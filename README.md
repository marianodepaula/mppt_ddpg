# mppt_ddpg
An implementation of DDPG for maximum power point traking of a photovoltaic system.
## Requirements: 

- Tensorflow 1
- numpy
- python 3.5
- Gym Open AI
- Gym environment 'mppt-v0'

### Source: mppt-gym
Clone the environment mmppt-gym from https://github.com/loavila/mppt-gym and install following the instructions.

It is recommendable installing in a python virtual environment (https://rukbottoland.com/blog/tutorial-de-python-virtualenv/)

#### Environment descriptions

- mmpt-v0 is a standard environment of a photovoltaic system (put link to the model)

- mmpt_shaded-v0 is an environment of a photovoltaic system with partial shading (put link to the model)


## How to run the training:
In a console run:

``` 
python main.py

```

### Testing:

``` 
python simul_test2.py

```



