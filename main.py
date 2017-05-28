import gym
import numpy as np
import random
from statistics import mean
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
no_of_games =  10000
training_data = []
game_mem = []
LR = 1e-4
gamma = 0.99
decay_rate = 0.99

def test_game():
    while True:
        env.render()
        obs,t,d,info = env.step(elsenv.action_space.sample())
        if d:
            env.reset()

def initialize():

    scores = []
    temp_i = np.random.rand(4) *2 -1
    noise = 0.1
    best_reward = 0
    for n_games in range(no_of_games):
        #print(n_games)
        score = 0
        game_mem = []
        prev_observation = []
        obs = env.reset()
        
        new_temp = temp_i + (np.random.rand(4)*2 - 1)*noise
        env.render()
        for _ in range(goal_steps):
            #print("starting new game step")
            action = 0 if np.matmul(new_temp, obs) < 0 else 1
            obs, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_mem.append([prev_observation, action])
            prev_observation = obs
            score += reward
            if done: break
        
        if score > best_reward:
            best_reward = score
            temp_i = new_temp

        if score >= 50:
            scores.append(score)
            for t in game_mem:
                if t[1] == 1:
                    output = [0,1]
                else:
                    output = [1,0]

                training_data.append([t[0], output])

        env.reset()

    total_training_data = np.array(training_data)
    np.save("trained_data.npy", training_data)
    print(len(training_data))
    #print("Average saved scores",mean(scores))
    #print(scores)
    return training_data

def neural_net(input_size):

    layer = input_data(shape=[None, input_size, 1], name = 'input')
    
    layer = fully_connected(layer, 128, activation = 'relu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 256, activation = 'relu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 512, activation = 'relu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 256, activation = 'relu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 128, activation = 'relu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 2, activation = 'softmax')

    layer = regression(layer, optimizer= 'adam', learning_rate = LR, loss = 'categorical_crossentropy',name = 'targets')
    model = tflearn.DNN(layer, tensorboard_dir = 'log')

    return model

def discount_reward(rew):
	discounted  = np.zeros_like(rew)
	added_rewards = 0
	for i in reversed(range(0,rew.size)):
		
		if rew[i] != 0 : added_rewards = 0    #need to change for cartpole
		added_rewards = added_rewards*gamma + rew[i]
		discounted[i] = added_rewards
		
	return discounted
	
def train_model(training_data):

    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    print("len of x", len(x))
    print("len of x[0]", len(x[0]))

    model = neural_net(len(x[0]))
    model.fit(x, y, n_epoch = 3, snapshot_step = 500, show_metric = True, run_id = 'game_learning')
    return model


training_data = initialize()
model = train_model(training_data)
model.save('saved_model.model')

#model = neural_net(len(training_data[0][0]))
#model.load('saved_model.model')

scores = []

for game in range(100):
    score = 0
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        obs, rew, done, info = env.step(action)
        #print("the rewards",rew)
        prev_obs = obs
        score += rew
        if done: break

    scores.append(score)

print('Avg score',sum(scores)/len(scores))