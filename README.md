# Stochastic-Synapses

Implementation of model proposed in the paper `Task Agnostic Continual Learning via Stochastic Synapses`. Please note that I am not the author of the paper and this is just an implementation of the model proposed by the authors Simon Schug, et al. The citation for the paper will be added once it is out. This paper was published at the ICML 2020 workshop for Continual Learning.

The link to the paper can be found [here](https://drive.google.com/file/d/1ZQg7Lb8IoVdHet3AwQFWTvg5QBUsbZJX/view?usp=sharing). 

Instructions to run the model:

1.  First, install the requirements using 

`pip install -r requirements.txt`

2.  Change the required parameters in the `parameters.py`* file.

3. Run either the `modified_network.py` or `stochastic_network.py` using 
`python3 stochastic_network.py`. 

* Note - For understanding the parameters please read the paper available [here](https://drive.google.com/file/d/1ZQg7Lb8IoVdHet3AwQFWTvg5QBUsbZJX/view?usp=sharing)

The current citation for the paper-
```
@misc{CLICMLAc42:online,
author = {Schug. Simon, Benzing. Frederik, Steger. Angelika},
title = {Task Agnostic Continual Learning via Stochastic Synapses},
howpublished = {\url{https://sites.google.com/view/cl-icml/accepted-papers?authuser=0}},
month = {},
year = {},
note = {(Accessed on 09/21/2020)}
}
```
