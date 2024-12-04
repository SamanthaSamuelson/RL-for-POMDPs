import gym
import tensorflow as tf
from tensorflow.keras import Model, losses, models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input, Add, Activation, Lambda, Concatenate, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import numpy as np
import xml.etree.ElementTree
import os
import pickle
import math

tf.keras.backend.set_floatx('float64')

'''
Deep deterministic policy gradient (actor-critic method) on Mujoco environment InvertedPendulumv4.  
The DDPG agent consists of actor and critic networks as well as train and evaluate methods.   

In this file we've updated the reward function to give more useful feedback, which allows for better training, and better distiction between policies.  
    Instead of giving a reward of +1 for every interaction with the environment in which the pendulum remains upright, we return pi/2-q, where q is the angle 
    of the pendulum measured from verticle.  This gives a larger reward the closer to verticle the pendulum remains.  This provides a numerical difference between policies which 
    both succeed in balancing the pendulum for 1000 steps, but have different degrees of "wobbliness".  

In this file, we include the possibility of three types of disturbance:
    1. Model inaccuracy or changing model: code allows for the possibility of changing the pendulum length, either while training or while testing the policy
        In this file it is possible to set the length of the inverted pendulum: standard length is 1, we vary lengths from .8 to 1.8.
    2. Observation noise: we can add Gaussian zero-mean noise to state observations while training and/or evaluation a policy
    3. Partial observability: The state space is four dimintional: linear cart position, angular pole position, linear cart velocity, angular pole velocity
        By truncating the state observation to 2 or three observations (out of 4) we make the problem partially obervable.  

By experimenting with these sources of error with the updated cost function, we can gain a sense of how various model imperfections cost the expected value of the problem
'''

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



class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.storage_idx = 0
        self.total_count = 0

        self.prev_state_buffer = np.zeros([self.max_size, self.state_size])
        self.next_state_buffer = np.zeros([self.max_size, self.state_size])
        self.action_buffer = np.zeros([self.max_size, self.action_size])
        self.reward_buffer = np.zeros([self.max_size, 1])
        self.dones_buffer = np.zeros([self.max_size, 1])

    def save_sample(self, observation_tuple):
        # order: state, action, reward, next state, done
        self.storage_idx = (self.total_count) % self.max_size
        self.prev_state_buffer[self.storage_idx] = observation_tuple[0]
        self.action_buffer[self.storage_idx] = observation_tuple[1]
        self.reward_buffer[self.storage_idx] = observation_tuple[2]
        self.next_state_buffer[self.storage_idx] = observation_tuple[3]
        self.dones_buffer[self.storage_idx] = observation_tuple[4]

        self.total_count += 1


    def get_training_batch(self): 
        current_size = np.min([self.total_count, self.max_size])
        sample_indices = np.random.choice(current_size, self.batch_size, replace=True) # assumes you dont sample till buffer is full
        prev_state_batch = tf.convert_to_tensor(self.prev_state_buffer[sample_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[sample_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[sample_indices])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[sample_indices])

        # order: state, action, reward, next state, done
        #buffers_tuple = [prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch]
        #return buffers_tuple
        return prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch






class DDPG_agent:
    def __init__(self, env):
        self.num_states = env.observation_space.shape[0] # gives integer
        # We can force partial observability by only observing first two states (positions)
        self.num_states = 3
        #self.state_space = env.observation_space.shape # gives tuple
        self.state_space = (self.num_states,)
        print('state space is ', self.state_space)
        print('potential new state space is ', (self.num_states,))
        print("Size of State Space ->  {}".format(self.num_states))
        self.num_actions = env.action_space.shape[0]
        self.action_space = env.action_space.shape # gives tuple
        print("Size of Action Space ->  {}".format(self.num_actions))

        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        low = env.action_space.low
        high = env.action_space.high
        self.action_bias = (high + low) / 2.
        self.action_multiplier = high - self.action_bias
        def clamper(actions):
            return np.clip(actions, a_max=env.action_space.high, a_min=env.action_space.low)
        self.clamper = clamper

        print("Max Value of Action ->  {}".format(self.upper_bound))
        print("Min Value of Action ->  {}".format(self.lower_bound))

        self.act_std_dev = 0.1
        self.noise_decay = 0.99
        self.obs_std_dev = 0.0
        #self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.act_std_dev) * np.ones(1))
        #self.noise = NoiseObject(0, self.act_std_dev)

        # Learning rate for actor-critic models
        # Discount factor for future rewards
        self.gamma = 0.9995
        # Used to update target networks
        self.tau = 0.002
        self.critic_lr = 0.001
        self.actor_lr = 0.001
        self.critic_optimizer = Adam(self.critic_lr)
        #self.critic_optimizer = RMSprop()
        self.actor_optimizer = Adam(self.actor_lr)
        #self.actor_optimizer = RMSprop()

        self.actor_model = self.create_actor_network()
        self.critic_model = self.create_critic_network()

        self.target_actor = self.create_actor_network()
        self.target_critic = self.create_critic_network()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())


        #self.total_episodes = num_training_episodes
        self.buffer_size = 100000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.num_states, self.num_actions)



    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_model, model, tau):
        weights = []
        targets = target_model.weights
        for i, weight in enumerate(model.weights):
            weights.append(weight * self.tau + targets[i] * (1 - self.tau))
        target_model.set_weights(weights)

    def create_actor_network(self):
        # input state, output action
        # note: if we use dropout or batch normalization use the training=TRUE (or false) flag when calling the model
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = Input(shape=self.state_space) 
        i = state_input
        #i = Flatten()(i)
        #i = Dense(32, activation="relu")(i)
        #i = Dense(32, activation="relu")(i)
        #i = Dense(64, activation="relu")(i)
        i = resdense(64)(i)
        #i = resdense(self.outputdims)(i)
        i = resdense(64)(i)
        i = resdense(32)(i)
        #i = resdense(256)(i)
        #i = resdense(128)(i)
        #i = resdense(64)(i)
        # map into (0,1)
        i = Dense(self.num_actions, kernel_initializer=last_init)(i)
        i = Activation('tanh')(i) # maps things to -1, 1
        # i = Activation('linear')(i)
        # map into action_space
        i = Lambda(lambda x: x * self.action_multiplier + self.action_bias)(i)

        out = i
        model = Model(inputs=[state_input], outputs=out)

        def my_loss_fn(_, crit_of_pred):
            return -tf.reduce_mean(crit_of_pred)  # Note the `axis=-1`
        model.compile(loss=my_loss_fn, optimizer=self.actor_optimizer)
        return model

    def create_critic_network(self):
        # input state, action, output value
        state_input = Input(shape=self.state_space) 
        # h = Dense(8, activation="relu")(state_input)
        #h = Dense(32, activation="relu")(h)

        action_input = Input(shape=self.action_space)
        # j = Dense(8, activation="relu")(action_input)

        #merged = Add()([h,j])
        #merged = Concatenate()([h,j])
        merged = Concatenate()([state_input, action_input])
        #i = Dense(64, activation="tanh")(merged)
        #i = Dense(64, activation="tanh")(i)
        #i = Dense(64, activation="tanh")(i)
        i = resdense(64)(merged)
        i = resdense(64)(i)
        i = resdense(32)(i)
        #i = resdense(128)(i)
        #i = resdense(64)(i)

        i = Dense(1)(i)
        #i = Activation('relu')(i)
        out = i

        model = Model(inputs=[state_input, action_input], outputs=out)
        model.compile(loss='mse', optimizer=self.critic_optimizer)

        return model



    def get_action(self, state, noisy=True):
        state = np.reshape(state, (1, self.num_states))
        sample_action = self.actor_model(state) # model(state) vs model.predict(state) does not matter here
        # noise = self.ou_noise()
        if noisy: noise = np.random.normal(0, self.act_std_dev, 1)
        else: noise = 0
        noisy_action = sample_action + noise
        legal_action = self.clamper(noisy_action)

        return legal_action[0]
        # pay attention - this may be an environment thing

#    Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        # Training and updating Actor & Critic networks.
        print('update function being called!')

        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + (1 - dones_batch)*self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            #actions = self.actor_model(next_state_batch, training=True)
            #critic_value = self.critic_model([next_state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            print('actor loss for updating: ', actor_loss, '\n')
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))


    def train(self, env, total_episodes=1000, render=False, dynamic_model=False, xml_path='', model_name='DDPG_model'):
        total_steps = 0
        change_idx = np.random.choice(range(50000,60000),1)
        print('model should change at ', change_idx)

        ep_reward_list = []
        ep_steps_list = []
        avg_reward_list = []
        max_reward = -np.inf

        for ep in range(total_episodes):

            prev_state = env.reset()[0]
            prev_state = prev_state[0:self.num_states] + np.random.normal(0, self.obs_std_dev, self.num_states)
            episodic_reward = 0
            steps = 0

            while True:
                if render: env.render()

                action = self.get_action(prev_state)
                state, _, done, _, _ = env.step(action)
                ## Here add option for observation noise or truncated observations
                state = state[0:self.num_states] + np.random.normal(0, self.obs_std_dev, self.num_states)
                reward = math.pi/2-state[1]
                steps += 1

                # order: state, action, reward, next state, done
                self.buffer.save_sample((prev_state, action, reward, state, done))
                episodic_reward += reward

                prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch = my_agent.buffer.get_training_batch()
                #print('calling update function')
                self.update(prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch)

                self.update_target(self.target_actor, self.actor_model, self.tau)
                self.update_target(self.target_critic, self.critic_model, self.tau)

                if done or steps > 1000:
                    break

                prev_state = state
                self.act_std_dev = self.act_std_dev * self.noise_decay

                total_steps += 1
                if (total_steps == change_idx) and (dynamic_model == True):
                    print('total steps are:', total_steps, 'changing model!')
                    total_steps = 0
                    change_idx = np.random.choice(range(1000,5000), 1)
                    env = self.change_model(env, xml_path, .6)
                    print('new change index:', change_idx, 'total steps at:', total_steps)

            ep_reward_list.append(episodic_reward)
            ep_steps_list.append(steps)
            avg_reward = np.mean(ep_reward_list[-10:])
            avg_steps = np.mean(ep_steps_list[-10:])
            avg_reward_list.append(avg_reward)
            #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            if (ep % 10 == 0):
                print('Episode', ep, 'average reward is:', avg_reward)
                print('Episode', ep, 'average steps :', avg_steps)
            if avg_reward >= max_reward:
                #self.actor_model.save(os.path.join("models", model_name))
                self.my_save(os.path.join("Good_models", model_name))
                max_reward = avg_reward
                print('saving run with average reward ', avg_reward)

            if (ep % 100 == 0): self.evaluate(env, episodes = 1)

    def change_model(self, env, xml_path, new_pend_length):
        et = xml.etree.ElementTree.parse(xml_path)
        print('xml object', et)
        root = et.getroot()
        for child in root:
            print(child.tag, child.attrib)

        pendulum_geom_lengths = root[4][1][2][1].get('fromto')
        pendulum_geom_density = root[4][1][2][1].get('density')
        print('current pole length', pendulum_geom_lengths)
        print('current pole density', pendulum_geom_density)

        #old_state = env.sim.get_state()
        old_state = env.data
        #print('all state info:', old_state.time, old_state.qpos, old_state.qvel, old_state.act,
        #      old_state.udd_state)
        print('state info', old_state)


        print('CHANGING PARAMETERS NOW')

        #new_pend_length = .6
        pendulum_geom_lengths = '0 0 0 0.001 0 {}'.format(new_pend_length)
        #pendulum_geom_lengths = '0 0 0 0.001 0 0.6'
        print('new pendulum (half) length .6')
        new_pend_density = 1000
        pendulum_geom_density = '{}'.format(new_pend_density)
        #pendulum_geom_density = '1000'
        print('new pendulum density', pendulum_geom_density)
        root[4][1][2][1].set('fromto', pendulum_geom_lengths)
        root[4][1][2][1].set('density', pendulum_geom_density)
        print('trying to access pole length', root[4][1][2][1].attrib)
        print('trying to access pole density', root[4][1][2][1].attrib)

        et.write(xml_model_fullpath)
        print('xml file has been written')

        print('re-make env - do we need to return this??')
        env = gym.make("InvertedPendulum-v4")
        env.reset()

        print('resetting model to previous state', env.set_state(old_state.qpos, old_state.qvel))
        env.set_state(old_state.qpos, old_state.qvel)
        print('Done! \n')
        return env

    def evaluate(self, env, episodes=10, render=False):
        ep_reward_list = []
        ep_steps_list = []
        for ep in range(episodes):
            episode_reward = 0
            steps = 0
            state = env.reset()[0]
            state = state[0:self.num_states] + np.random.normal(0, self.obs_std_dev, self.num_states)
            while True:
                if render: env.render()

                action = self.get_action(state, noisy=False)
                state, _, done, _, _ = env.step(action)
                state = state[0:self.num_states] + np.random.normal(0, self.obs_std_dev, self.num_states)
                reward = math.pi / 2 - state[1]
                steps += 1
                episode_reward += reward
                if done or steps > 1000:
                    break

            ep_reward_list.append(episode_reward)
            ep_steps_list.append(steps)
            print('EVALUATION RUN episode: ', ep, 'reward:', episode_reward, 'steps: ', steps)

    def my_save(self, name):
        W_a = self.actor_model.get_weights()
        W_c = self.critic_model
        pickle.dump(W_a, open(name, 'wb'))
        #print('saving!')
        # print(W[0])

    def my_load(self, name):
        W_a = pickle.load(open(name, 'rb'))
        self.actor_model.set_weights(W_a)
        # print(W[0])





environment = gym.make("InvertedPendulum-v4")

model_path = 'inverted_pendulum.xml'
if model_path.startswith("/"):
    xml_model_fullpath = model_path
else:
    xml_model_fullpath = os.path.join(os.path.dirname(__file__), ".venv/lib/site-packages/gym/envs/mujoco/assets", model_path)



name = 'InvPendv4_PartialObs_2'
length = .5

my_agent = DDPG_agent(environment)
environment = my_agent.change_model(environment, xml_model_fullpath, length)
#my_agent.my_load(os.path.join("models", name))
my_agent.train(environment,total_episodes=5000, dynamic_model=False, xml_path=xml_model_fullpath, model_name=name)
#my_agent.my_save(os.path.join("Good_models", name))
my_agent.evaluate(environment, episodes=20)
#my_agent.my_load(name)
#my_agent.actor_model = models.load_model(os.path.join("models", name))
#my_agent.evaluate(environment, episodes=20)
# my_agent.my_save(name)



