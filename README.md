# Friday

Everyone love friday! Friday is a set of toolkit to build virtual assistants and serve it as a service for different frontier platforms like, Telegram, Instagram, Facebook messager. The ultimate aim is to build an agent that can understand human instructions and learn interactivly. That's the virtual assistant friday from Tony Stark!  

## What is Friday?
Friday is a toolkit for building general purpose agents for usages in real-life. Agents built with friday toolkit is a genral interface for people to interact and do tasks for them.

## Compositional agent
A Compositional agent is composed of different operating units. It breaks a agent into different units, like sensors, dialog_adaptor, fulfillment router, fulfillment adaptor.

### Sensors
A `Sensor` is the primary data processing layer of an agent. Different sensors process different types of inputs. Outputs from different sensors can be integrated.

### fulfillment_adaptor()
fulfillment_adaptor() is a method that convert the output from all sensors to key and fulfillment arguments.

### FulfillmentRouter()
fulfillment_router accepts keys and run the corresponding fulfillment_engine object (fulfillment_engine.run(*args, **kwargs))

### FulfillmentEngine
fulfillment_engine handles running a specific task and return the results to the agent.

### dialog_adaptor()
This method aggregate sensor outputs, routing and fulfillment return and prdocue a nice response to the user.


### examples
```python

obs = {"text": "How are you?"}

output = agent.act(obs)  # use act() to ask the agent to work

```



## Setup environment
Develop on ubunut 18.04
1. Install packages
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
```
2. Install miniconda for creating isolated python environment
3. Install PyTorch>=1.7.1


## Requirements
```bash
pip install -r requirements.txt
```
