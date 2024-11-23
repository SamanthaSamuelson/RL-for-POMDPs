#Source: https://github.com/AdamStelmaszczyk/rl-tutorial

import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Input, Dropout, Add
from keras.optimizers import Adam, SGD

#env = gym.make("InvertedDoublePendulum-v2")
#env = gym.make("InvertedPendulum-v4")
#env = gym.make('Pendulum-v0')
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
#env = gym.make('Acrobot-v1')

env._max_episode_steps = 500
reward_offset = 0

#name = 'DQ_learning_MountainCar'
#name = 'Good_models\DQ_acrobot_try1_resdense'
name = 'DQ_cartPole-v2.keras'
# name = 'DQ_acrobot_try2'
# name = 'DQ_invDoublePend_resDense'
#name = 'DQ_resdense_Pendn0'

#env._max_episode_steps = 500


def explore_env(env):
    print("State space:", env.observation_space)
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)
    print('state space shape', env.observation_space.shape)
    print('state space dim', len(env.observation_space.low))
    num_states = len(env.observation_space.low)
    # Explore action space
    print("Action space:", env.action_space)
    #print("- low:", env.action_space.low)
    #print("- high:", env.action_space.high)
    #dim_actions = len(env.action_space.low)

    observation = env.reset()
    #for t in range(500):
    #    env.render()
    #    action = env.action_space.sample()
    #    observation, reward, done, info = env.step(action)
    #env.close()


def resdense(features):
    def unit(i):
        hfeatures = max(4, int(features / 4))

        ident = i
        i = Dense(features, activation='tanh')(i)

        ident = Dense(hfeatures)(ident)
        ident = Dense(features)(ident)

        added = Add()([ident, i])

        # return merge([ident,i],mode='sum')
        return added

    return unit

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, observation, action, reward, next_obs, done):
        data = (observation, action, reward, next_obs, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def _encode_sample(self, indices):
        goals, observations, actions, rewards, next_observations, dones = [], [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            observation, action, reward, next_obs, done = data
            observations.append(np.array(observation, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_observations.append(np.array(next_obs, copy=False))
            dones.append(done)
        return np.array(observations), np.array(actions), np.array(rewards), np.array(next_observations), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        #print('indexs:', idxes)
        return self._encode_sample(idxes)

class DQAgent:
    def __init__(self, env):
        self.REPLAY_BUFFER_SIZE = 50000
        self.memory = ReplayBuffer(self.REPLAY_BUFFER_SIZE)
        self.eval_episodes = 100
        self.TRAIN_START = 1000
        self.TARGET_UPDATE_EVERY = 500
        self.EVAL_EVERY = 500
        self.batch_size = 64
        self.gamma = 1  # discount rate
        self.epsilon = .8  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.02
        self.current_performance = 0
        self.observation_stack_factor = 1

        #self.n_actions = 5
        self.n_actions = env.action_space.n
        print('n_actions', self.n_actions)
        #self.action_list = np.linspace(env.action_space.low, env.action_space.high, self.n_actions)
        self.action_list = np.array([0,1,2])
        print('action list: ', self.action_list)


        self.dim_state = len(env.observation_space.low)
        self.state_shape = env.observation_space.shape
        print('self state shape =', self.state_shape)
        self.state_size = env.observation_space.shape[0]
        print('self state size =', self.state_size)
        # self.state_size = 2
        self.inputdims = self.state_size * self.observation_stack_factor
        print('inputdims =', self.inputdims)
        self.model = self._build_model()
        self.model.summary()
        self.target_model = self._build_model()

    def one_hot_encode(self, action):
        idx = (np.abs(self.action_list - action)).argmin()
        one_hot = np.zeros(self.n_actions)
        one_hot[idx] = 1
        return one_hot

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        observations_input = Input(self.state_shape, name='observations_input')
        action_mask = Input((self.n_actions,), name='action_mask')
        i = observations_input
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(64)(i)
        hidden_4 = Dense(128, activation='relu')(i)
        output = Dense(self.n_actions, activation='linear')(hidden_4)
        filtered_output = keras.layers.multiply([output, action_mask])
        model = keras.models.Model([observations_input, action_mask], filtered_output)
        optimizer = Adam(learning_rate:=self.learning_rate)
        sgd = keras.optimizers.SGD(learning_rate:=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer, loss='mean_squared_error')
        print("model input shape: ")
        print(model.input_shape)
        return model


    def my_save(self, name):
        print('saving model:')
        self.model.save(name)

    def load_model(self, model_filename):
        self.model = load_model(model_filename)
        print('Loaded {}'.format(model_filename))
        self.model.summary()

    def my_predict(self, observations):
        #print('length of observations:', len(observations))
        action_mask = np.ones((len(observations), self.n_actions))
        #print('observation: ', observations)
        newObs = np.array([observations[0]])
        #print('new obs', newObs)
        #print('new obs shape', newObs.shape)
        #print('action mask: ', action_mask)
        #print('action mask shape: ', action_mask.shape)
        return self.model.predict(x=[newObs, action_mask],verbose=None)


    def target_predict(self, observations):
        #print('target predict observations shape', observations.shape)
        action_mask = np.ones((len(observations), self.n_actions))
        #print('intput to target predict ', [observations, action_mask])
        return self.target_model.predict(x=[observations, action_mask],verbose=None)

    def take_greedy_action(self, observation):
        #print('observation input: ', observation)
        next_q_values = self.my_predict(observations=[observation])
        #print('next q values:', next_q_values)
        foo = np.argmax(next_q_values)
        #print('index of optimal action: ', foo)
        return self.action_list[foo]

    def take_epsilon_greedy_action(self, observation, epsilon):
        #print('observation input: ', observation)
        if random.random() < epsilon:
            action = env.action_space.sample()
            #print('took random action ', action)
        else:
            action = self.take_greedy_action(observation)
            #print('took optimal action ', action)
        return action

    def evaluate_current_policy(self, view):
        #print("Evaluation")
        episode_return = 0.0
        episode_duration_avg = 0
        episode_return_avg = 0
        episode_return_list = np.zeros(self.eval_episodes)
        episode_duration_list = np.zeros(self.eval_episodes)
        num_actions_taken = 0
        for ep in range(self.eval_episodes):
            observation = env.reset()
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            reward += reward_offset
            episode_return += reward
            observation = next_obs
            while True:
                if view: env.render()
                #next_q_values = self.my_predict(observations=[observation])
                #print('current state:', observation)
                #print('next q values:', next_q_values)
                action = self.take_greedy_action(observation)
                #print('took action: ', action)
                num_actions_taken +=1
                observation, reward, done, _, _ = env.step(action)
                reward += reward_offset
                episode_return += reward
                if done:
                    episode_return_list[ep] = episode_return
                    episode_duration_list[ep] = num_actions_taken
                    #print('total return ', episode_return, 'matches actions taken', num_actions_taken)
                    episode_return = 0.0
                    num_actions_taken = 0
                    #print('and now were done \n')
                    break
        #print('total episode return list:', episode_return_list)
        episode_return_avg = np.average(episode_return_list)
        episode_duration_avg = np.average(episode_duration_list)
        print('average length of episode', episode_duration_avg)
        #print('sanity check:')
        #print(self.my_predict(observations=[[0,0,0,0]]))
        #print(self.my_predict(observations=[[0,0,2,-2]]))
        #print(self.my_predict(observations=[[1, 0, 0]]))
        #print(self.my_predict(observations=[[0,1,0]]))
        #print(self.my_predict(observations=[[0,0]]))
        if episode_return_avg > self.current_performance:
            self.current_performance = episode_return_avg
            agent.my_save(name)
        return episode_return_avg

    def demonstrate_policy(self, view, dem_length):
        print("Demonstration")
        episode_return = 0.0
        episode_return_list = np.zeros(dem_length)
        episode_duration_list = np.zeros(dem_length)
        num_actions_taken = 0
        for episode in range(dem_length):
            observation = env.reset()
            while True:
                if view: env.render()
                action = self.take_greedy_action(observation)
                num_actions_taken += 1
                observation, reward, done, info = env.step(action)
                reward += reward_offset
                episode_return += reward
                if done:
                    episode_return_list[episode] = episode_return
                    episode_duration_list[episode] = num_actions_taken
                    episode_return = 0.0
                    num_actions_taken = 0
                    break
        print('length of episodes', episode_duration_list)
        print('reward of episodes', episode_return_list)

    def fit_batch(self, batch):
        observations, actions, rewards, next_observations, dones = batch
        # Predict the Q values of the next states. Passing ones as the action mask.
        next_q_values = self.target_predict(next_observations)
        # The Q values of terminal states is 0 by definition.
        next_q_values[dones] = 0.0
        #print('q values of terminal states', next_q_values[dones])
        # The Q values of each start state is the reward + gamma * the max next state Q value
        q_values = rewards + self.gamma * np.max(next_q_values, axis=1)
        one_hot_actions = np.array([self.one_hot_encode(action) for action in actions])
        #train on state action pairs:
        x = [observations, one_hot_actions]
        #print('x train data shape:', len(x), len(x[0]), len(x[1]), len(x[0][0]), len(x[1][0]))
        # withthe "updated" q values as y values
        y = one_hot_actions * q_values[:, None]
        history = self.model.fit(x, y, batch_size=self.batch_size, verbose=0)
        return history.history['loss'][0]

    def train(self, max_steps):
        print('Training!')
        episode_return = 0.0
        for episode in range(max_steps):
            obs = env.reset()
            #print('done value??', done)
            #print('first observation: ', obs)
            # observations seem to be a different shape initially ... take on random action to start
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            reward += reward_offset
            episode_return += reward
            #self.memory.add(obs, action, reward, next_obs, done)
            obs = next_obs
            #print('observation: ', obs)

            while True:
                action = self.take_epsilon_greedy_action(obs,self.epsilon)
                next_obs, reward, done, _, _ = env.step(action)
                reward+=reward_offset
                episode_return += reward
                self.memory.add(obs, action, reward, next_obs, done)
                obs = next_obs
                #print('observation: ', obs)
                if done:
                    #print('episode num',episode,'got return',episode_return)
                        # for acrobot:
                    #if episode_return > -500: print('episode num',episode,'got return',episode_return)
                        # for cartpole:
                    if episode_return > 20: print('episode num', episode, 'got return', episode_return)
                    episode_return = 0.0
                    break

            if episode >= self.TRAIN_START:
                #print('episode in training phase')
                if episode % self.TARGET_UPDATE_EVERY == 0:
                    #print(episode % self.TARGET_UPDATE_EVERY)
                    self.target_model.set_weights(self.model.get_weights())
                    #print('target model updated to normal model')
                    self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
                    print('new epsilon: ', self.epsilon)
                #print('get batch')
                batch = self.memory.sample(self.batch_size)
                #print('fit batch')
                self.fit_batch(batch)
            if episode >= self.TRAIN_START and episode % self.EVAL_EVERY == 0:
                print('Evaluating!!')
                print('epsilon: ', self.epsilon)
                episode_return_avg = self.evaluate_current_policy(view=0)
                print('episode ', episode, 'episode_return_avg: ', episode_return_avg)





explore_env(env)
agent = DQAgent(env)
#agent.train(500000)
#agent.my_save(name)
agent.load_model(name)
env._max_episode_steps = 500
agent.evaluate_current_policy(view=0)
agent.demonstrate_policy(view=1, dem_length=5)

# TEST POLICY!
