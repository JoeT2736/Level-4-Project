##### Vicsek Model, constant speed, new direction = average of neighbours #####



import numpy as np

agent_position = np.
agent_direction = np.
v = np. #Distance travelled by each agent during time taken to update
range = #Range of interaction
nearest_neighbours = np. #Number of neighbours each agent interacts with
FOV = 
plane = #Size of plane agents move on
noise = 
vector_motion = #velocity of each agent
time_step = 


def movement(agent_position, agent_direction, v, plane):
    move = (np.cos(agent_direction), np.sin(agent_direction)) * v
    agent_position += move
    agent_position[(agent_position < 0)] += plane
    agent_position %= plane

    return agent_position, move


def New_Direction(pos, dir, eta, neighbour)
    