import gym
import tensorflow as tf
from tensorflow.keras import Model, losses, models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input, Add, Activation, Lambda, Concatenate, LSTM, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import numpy as np
import xml.etree.ElementTree
import os
import pickle
from sklearn.utils import shuffle
import random

tf.keras.backend.set_floatx('float64')

# OK FIRST TRY TO id MODEL WITH A PERFECTLY TRAINED ACTOR ALREADY IN PLACE
# for now we will keep density the same, change length

# !!!! FOR NOW DO NOT CHANGE ACTOR CRITIC MODEL SHAPE!!!!!

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
        self.params_buffer = np.zeros([self.max_size,1])

    def save_sample(self, observation_tuple):
        # order: state, action, reward, next state, done, param
        self.storage_idx = (self.total_count) % self.max_size
        self.prev_state_buffer[self.storage_idx] = observation_tuple[0]
        self.action_buffer[self.storage_idx] = observation_tuple[1]
        self.reward_buffer[self.storage_idx] = observation_tuple[2]
        self.next_state_buffer[self.storage_idx] = observation_tuple[3]
        self.dones_buffer[self.storage_idx] = observation_tuple[4]
        self.params_buffer[self.storage_idx] = observation_tuple[5]

        self.total_count += 1


    def get_training_batch(self): # should this be different for RNN?
        # move casting to tf here, to match source code better
        # order: state, action, reward, next state, done, param
        current_size = np.min([self.total_count, self.max_size])
        sample_indices = np.random.choice(current_size, self.batch_size, replace=True) # assumes you dont sample till buffer is full
        prev_state_batch = tf.convert_to_tensor(self.prev_state_buffer[sample_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[sample_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[sample_indices])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[sample_indices])
        params_batch = tf.convert_to_tensor(self.params_buffer[sample_indices])
        return prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch, params_batch






class DDPG_agent:
    def __init__(self, env):
        self.hist_length = 50

        self.num_states = env.observation_space.shape[0] # gives integer
        self.state_space = env.observation_space.shape # gives tuple
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

        self.std_dev = 0.1
        self.noise_decay = 0.999


        # Learning rate for actor-critic models
        # Discount factor for future rewards
        self.gamma = 0.9995
        # Used to update target networks
        self.tau = 0.002
        self.critic_lr = 0.001
        self.actor_lr = 0.0005
        self.critic_optimizer = Adam(self.critic_lr)
        #self.critic_optimizer = RMSprop()
        self.actor_optimizer = Adam(self.actor_lr)
        #self.actor_optimizer = RMSprop()
        self.optimizer = Adam()

        self.actor_model = self.create_actor_network()
        self.critic_model = self.create_critic_network()

        self.target_actor = self.create_actor_network()
        self.target_critic = self.create_critic_network()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.sys_ID_network = self.create_sysID_network_1()

        #self.total_episodes = num_training_episodes
        self.buffer_size = 100000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.num_states, self.num_actions)



    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_model, model):
        weights = []
        targets = target_model.weights
        for i, weight in enumerate(model.weights):
            weights.append(weight * self.tau + targets[i] * (1 - self.tau))
        target_model.set_weights(weights)


    def create_actor_network(self):
        # input state, output action
        # note: if we use dropout or batch normalization use the training=TRUE (or false) flag when calling the model
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = Input(shape=self.state_space)  ## WANT TO BE ABLE TO SET STATE_SPACE SIZE AUTOMATICALLY, NEED TO FIX
        i = state_input
        mu = Input(shape=(1,))
        #j = Dense(16)(mu)
        i = Concatenate()([i,mu])
        i = resdense(64)(i)
        #i = resdense(16)(i)
        #i = Concatenate()([i,j])
        i = resdense(32)(i)
        i = Dense(self.num_actions, kernel_initializer=last_init)(i)
        i = Activation('tanh')(i) # maps things to -1, 1
        i = Lambda(lambda x: x * self.action_multiplier + self.action_bias)(i)

        out = i
        model = Model(inputs=[state_input,mu], outputs=out)

        def my_loss_fn(_, crit_of_pred):
            return -tf.reduce_mean(crit_of_pred)  # Note the `axis=-1`
        #model.compile(loss=my_loss_fn, optimizer=self.actor_optimizer)
        # source doesn't compile, so for now we dont either
        return model
        # ok, mean squared error is really the wrong loss here - we dont use it anyway


    def create_critic_network(self):
        # input state, action, output value
        state_input = Input(shape=self.state_space) ## WANT TO BE ABLE TO SET STATE_SPACE SIZE AUTOMATICALLY, NEED TO FIX
        action_input = Input(shape=self.action_space)
        mu = Input(shape=(1,))

        merged = Concatenate()([state_input, action_input,mu])
        i = resdense(64)(merged)
        i = resdense(64)(i)
        #i = resdense(32)(i)
        i = Dense(1)(i)
        #i = Activation('relu')(i)
        out = i

        model = Model(inputs=[state_input, action_input, mu], outputs=out)
        #model.compile(loss='mse', optimizer=self.critic_optimizer)

        return model

    def create_sysID_network_1(self):
        # how to best input HISTORIES??
            # try inputting long histories, vs LSTM .... does it need to learn on sequences??
        # start by training it on sequences of 1000, then decrease?
        # how to feed data in correct shape?
        DROPOUT = .1

        prev_state_input = Input(shape=(self.hist_length, self.num_states,))
        h1 = Dense(8, activation="relu")(prev_state_input)
        action_input = Input(shape=(self.hist_length, self.num_actions,))
        h2 = Dense(8, activation="relu")(action_input)
        #next_state_input = Input(shape=(self.hist_length, self.num_states))
        #h3 = Dense(8, activation="relu")(next_state_input)

        i =Concatenate()([h1,h2])
        i = Flatten()(i)
        i = Dense(256, activation="tanh")(i)
        i = Dense(128, activation="tanh")(i)
        i = Dense(64, activation="tanh")(i)

        # ultimately this needs to output a model parameter ...
        i = Dense(1)(i)
        out = i

        model = Model(inputs=[prev_state_input,action_input], outputs=out)
        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def train_sysID_network(self, s_data, a_data, y_data, name):
        print('data shapes:', np.shape(s_data))
        print('targets shape:', np.shape(y_data))
        print('why is the output this weird shape?')
        #foo = self.sys_ID_network.predict([[s_data[0]], [a_data[0]]])
        #print(foo)
        #print(len(foo[0]))
        #print('foo shape', np.shape(foo))
        print('state data dhape', np.shape(s_data))
        print('action data shape', np.shape(a_data))
        s_data, a_data, y_data = shuffle(s_data, a_data, y_data)
        self.sys_ID_network.fit([s_data,a_data], y_data, epochs=10, validation_split = .1)
                            #    validation_data = ([s_ver, a_ver], y_ver) , validation_steps=5 )

        #self.my_save_sysID(name)
        self.my_save_sysID(os.path.join("models", name))




    def get_action(self, state, mu, noisy=True):
        state = np.reshape(state, (1, self.num_states))
        mu = np.reshape(mu, (1,1))
        sample_action = self.actor_model([state,mu]) # model(state) vs model.predict(state) does not matter here
        # noise = self.ou_noise()
        if noisy: noise = np.random.normal(0, self.std_dev, 1)
        else: noise = 0
        noisy_action = sample_action + noise
        legal_action = self.clamper(noisy_action)

        return legal_action[0]
        # pay attention - this may be an environment thing

#    Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch, params_batch):
        # Training and updating Actor & Critic networks.
        print('update function being called!')
        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor([next_state_batch,params_batch], training=True)
            y = reward_batch + (1 - dones_batch)*self.gamma * self.target_critic([next_state_batch, target_actions, params_batch], training=True)
            critic_value = self.critic_model([state_batch, action_batch, params_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor_model([state_batch, params_batch], training=True)
            critic_value = self.critic_model([state_batch, actions, params_batch], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))


    def train_AC(self, env, total_episodes=1000, render=False, dynamic_model=False, xml_path='', model_name='DDPG_model'):
        total_steps = 0
        change_idx = np.random.choice(range(50000,60000),1)
        if(dynamic_model==True): print('model should change at ', change_idx)
        length =.6
        env = self.change_model(env, xml_path, length)

        ep_reward_list = []
        avg_reward_list = []
        max_reward = -np.inf

        for ep in range(total_episodes):
            length = np.round(np.random.uniform(.3, 1, 1), 2)[0]
            env = self.change_model(env, xml_path, length)
            prev_state = env.reset()[0]
            episodic_reward = 0

            while True:
                if render: env.render()

                action = self.get_action(prev_state,length)
                state, reward, done, _, _ = env.step(action)

                # order: state, action, reward, next state, done
                self.buffer.save_sample((prev_state, action, reward, state, done, length))
                episodic_reward += reward

                prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch, params_batch\
                    = my_agent.buffer.get_training_batch()
                self.update(prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch, params_batch)

                self.update_target(self.target_actor, self.actor_model)
                self.update_target(self.target_critic, self.critic_model)

                if done:
                    break

                prev_state = state
                self.std_dev = self.std_dev*self.noise_decay

                total_steps += 1
                if (total_steps == change_idx) and (dynamic_model == True):
                    print('total steps are:', total_steps, 'changing model!')
                    total_steps = 0
                    change_idx = np.random.choice(range(1000,5000), 1)
                    env = self.change_model(env, xml_path)
                    print('new change index:', change_idx, 'total steps at:', total_steps)

            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[-10:])
            avg_reward_list.append(avg_reward)
            if (ep % 10 == 0):
                print('Episode', ep, 'average reward is:', avg_reward)
            if avg_reward >= max_reward:
                self.my_save_actor(model_name)
                max_reward = avg_reward

            if (ep % 100 == 0):
                length = np.round(np.random.uniform(.3, 1, 1), 2)[0]
                env = self.change_model(env, xml_path, length)
                self.evaluate(env, length, episodes = 1)

    def change_model(self, env, xml_path, new_pend_length):
        et = xml.etree.ElementTree.parse(xml_path)
        #print('xml object', et)
        root = et.getroot()
        for child in root:
            print(child.tag, child.attrib)

        pendulum_geom_lengths = root[4][1][2][1].get('fromto')
        pendulum_geom_density = root[4][1][2][1].get('density')
        #print('current pole length', pendulum_geom_lengths)
        #print('current pole density', pendulum_geom_density)

        # old_state = env.sim.get_state()
        old_state = env.data
        # print('all state info:', old_state.time, old_state.qpos, old_state.qvel, old_state.act,
        #      old_state.udd_state)
        #print('state info', old_state)

        #print('CHANGING PARAMETERS NOW')

        # new_pend_length = .6
        pendulum_geom_lengths = '0 0 0 0.001 0 {}'.format(new_pend_length)
        # pendulum_geom_lengths = '0 0 0 0.001 0 0.6'
        #print('new pendulum (half) length .6')
        new_pend_density = 1000
        pendulum_geom_density = '{}'.format(new_pend_density)
        # pendulum_geom_density = '1000'
        #print('new pendulum density', pendulum_geom_density)
        root[4][1][2][1].set('fromto', pendulum_geom_lengths)
        root[4][1][2][1].set('density', pendulum_geom_density)
        #print('trying to access pole length', root[4][1][2][1].attrib)
        #print('trying to access pole density', root[4][1][2][1].attrib)

        et.write(xml_model_fullpath)
        #print('xml file has been written')

        #print('re-make env - do we need to return this??')
        env = gym.make("InvertedPendulum-v4")
        env.reset()

        #print('resetting model to previous state', env.set_state(old_state.qpos, old_state.qvel))
        env.set_state(old_state.qpos, old_state.qvel)
        #print('Done! \n')
        return env


    def evaluate(self, env, length, episodes=10, render=False):
        print('Evaluating!')
        ep_reward_list = []
        for ep in range(episodes):
            episode_reward = 0
            steps = 0
            state = env.reset()[0]
            while True:
                if render: env.render()

                action = self.get_action(state, length, noisy=False)
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                steps += 1
                if done or steps > 1000:
                    break

            ep_reward_list.append(episode_reward)
            print('EVALUATION RUN episode: ', ep, 'length:', length, 'reward:', episode_reward)


    def online_sys_ID(self,env,xml_path,total_episodes=10000,render=False):
        ep_reward_list = []
        total_steps = 0
        #change_idx = np.random.choice(range(50, 500), 1)
        change_idx = 2000
        print('model should change at ', change_idx)

        for ep in range(total_episodes):
            length = np.round(np.random.uniform(.4, .9, 1), 2)[0]
            print("episode number", ep, "real pendulum length:", length)
            env = self.change_model(env, xml_path, length)
            episodic_reward = 0
            prev_state = env.reset()[0]
            idx = 0

            s1_data_point = np.zeros([self.hist_length, self.num_states])
            a_data_point = np.zeros([self.hist_length, self.num_actions])

            s1_data_point[idx] = prev_state
            action = env.action_space.sample()
            a_data_point[idx] = action

            while True:
                if render: env.render()

                y_pred = self.sys_ID_network.predict(
                    [[np.reshape(s1_data_point,(1,self.hist_length,self.num_states))],
                     [np.reshape(a_data_point,(1,self.hist_length,self.num_actions))]], steps=1, verbose=0)
                #print('actual length ', length, 'pridected length', y_pred)
                #action = self.get_action(prev_state, length)
                action = self.get_action(prev_state, y_pred)
                state, reward, done, _, _ = env.step(action)
                episodic_reward += reward

                s1_data_point = self.add_obs_to_window(s1_data_point, prev_state, idx)
                a_data_point = self.add_obs_to_window(a_data_point, action, idx)
                idx += 1

                if done or idx > 1000: break

                prev_state = state

                total_steps += 1
                if (total_steps == change_idx):
                    total_steps = 0
                    change_idx = np.random.choice(range(50, 500), 1)
                    length = np.round(np.random.uniform(.4, .9, 1), 2)[0]
                    print('total steps are:', total_steps, 'changing model to new length ', length)
                    env = self.change_model(env, xml_path, length)
                    #print('new change index:', change_idx, )
                    #print('current episode reward', episodic_reward)


            ep_reward_list.append(episodic_reward)
            print('EVALUATION RUN episode: ', ep+1, 'reward:', episodic_reward, 'true length ', length,
                  'final esimated length ', y_pred, '\n' )

    def generate_sysID_training_data(self,env, model_names_list, lengths_list, xml_path):
        # we would need data from several different pendulum lengths .

        episodes = 300
        episode_length = self.hist_length
        num_models = len(model_names_list)
        env.reset()

        s1_data = np.zeros([episodes*num_models, episode_length, self.num_states])
        a_data = np.zeros([episodes*num_models, episode_length, self.num_actions])
        y_data = np.zeros([episodes*num_models, 1])

        for i in range(len(model_names_list)):
            print('loading model', model_names_list[i])
            self.my_load_actor(model_names_list[i])

            # need to set environment to correct model params ot simulate ...
            # need a name to save to ... what data format should we save as ? picle.dump?
            print('setting pendulum length to', lengths_list[i])
            env = self.change_model(env, xml_path, lengths_list[i])

            s1_data_point = np.zeros([episode_length, self.num_states])
            a_data_point = np.zeros([episode_length, self.num_actions])
            # keep these seperate then zip
            # s2_data = np.zeros([episode_length, self.num_states])

            for ep in range(episodes):
                episode_reward = 0
                idx = 0
                state = env.reset()[0]
                while True:
                    s1_data_point[idx] = state
                    action = self.get_action(state, noisy=False)
                    a_data_point[idx] = action
                    state, reward, done, info, _ = env.step(action)
                    #s2_data[idx] = state
                    episode_reward += reward
                    if done or idx>1000: break
                    idx += 1
                print('total episode reward', episode_reward)
                s1_data[(i*episodes) + ep] = s1_data_point
                a_data[(i*episodes) + ep] = a_data_point
                y_data[(i*episodes) + ep] = lengths_list[i]
                print('storing data in index ', ((i*episodes) + ep))
            #print('data shape:', np.shape(s1_data))
            #pickle.dump([s1_data, a_data])

        return s1_data, a_data, y_data

    def generate_sysID__bad_training_data(self,env,xml_path):
        # we would need data from several different pendulum lengths .
        # ok two ways to generate non-perfect runs: a (single?) bad model, or random actions

        episodes = 10
        episode_length = self.hist_length
        #num_lengths = len(lengths_list)
        num_lengths = 5000
        env.reset()

        s1_data = np.zeros([episodes*num_lengths, episode_length, self.num_states])
        a_data = np.zeros([episodes*num_lengths, episode_length, self.num_actions])
        y_data = np.zeros([episodes*num_lengths, 1])

        for i in range(num_lengths):
            #print('loading model', model_names_list[i])
            #self.my_load_actor(model_names_list[i])

            # need to set environment to correct model params ot simulate ...
            # need a name to save to ... what data format should we save as ? picle.dump?
            # length = lengths_list[i]
            lngth = np.round(np.random.uniform(.6,.9,1),3)[0]
            #print('setting pendulum lenght to', lngth)
            env = self.change_model(env, xml_path, lngth)



            for ep in range(episodes):

                s1_data_point = np.zeros([episode_length, self.num_states])
                a_data_point = np.zeros([episode_length, self.num_actions])
                # keep these seperate then zip
                # s2_data = np.zeros([episode_length, self.num_states])

                episode_reward = 0
                idx = 0
                state = env.reset()[0]

                while True:
                    s1_data_point[idx] = state
                    #action = self.get_action(state, noisy=False)
                    action = env.action_space.sample()
                    a_data_point[idx] = action
                    state, reward, done, info, _ = env.step(action)
                    #s2_data[idx] = state
                    episode_reward += reward
                    if done or idx > 1000: break
                    idx += 1
                #print('total episode reward', episode_reward)
                s1_data[(i*episodes) + ep] = s1_data_point
                a_data[(i*episodes) + ep] = a_data_point
                y_data[(i*episodes) + ep] = lngth
                #print('storing data in index ', ((i*episodes) + ep))
            #print('data shape:', np.shape(s1_data))
            #pickle.dump([s1_data, a_data])

        return s1_data, a_data, y_data

    def generate_ok_training_data_from_iffy_model(self,env,xml_path,lengths_list,model_names_list,working_model_name,render=False,random=False):
        # need to load a working sysED model for this to work!!!!!
        self.sys_ID_network = models.load_model(os.path.join("models", working_model_name))

        ep_reward_list = []
        episodes = 1000
        episode_length = self.hist_length

        s1_data = np.zeros([1, episode_length, self.num_states])
        a_data = np.zeros([1, episode_length, self.num_actions])
        y_data = np.zeros([1, 1])

        for ep in range(episodes):
            length = np.round(np.random.uniform(.4, .9, 1), 2)[0]
            print("episode number", ep, "real pendulum length:", length)
            env = self.change_model(env, xml_path, length)
            episode_reward = 0
            state = env.reset()[0]
            idx = 0

            s1_data_point = np.zeros([self.hist_length, self.num_states])
            a_data_point = np.zeros([self.hist_length, self.num_actions])

            s1_data_point[idx] = state
            action = env.action_space.sample()
            a_data_point[idx] = action

            while True:
                if render: env.render()

                # print('step number', idx)
                y_pred = self.sys_ID_network.predict([[s1_data_point], [a_data_point]], steps=1)
                # print('actual length ',length,'pridected length', y_pred)
                model_name = self.get_best_controller(y_pred, lengths_list, model_names_list)
                # need to first id length, match to model - pad samples appropraitely - for now?
                # print('loading model', model_name)
                self.my_load_actor(model_name)

                # s1_data_point[idx] = state
                s1_data_point = self.add_obs_to_window(s1_data_point, state, idx)
                if random:
                    action = env.action_space.sample()
                else:
                    action = self.get_action(state, noisy=False)
                # a_data_point[idx] = action
                a_data_point = self.add_obs_to_window(a_data_point, action, idx)
                state, reward, done, info, _ = env.step(action)
                # s2_data[idx] = state
                episode_reward += reward
                if done or idx > 1000: break
                idx += 1

                if idx % 20 == 0:
                    #print('adding data at idx', idx)
                    #print('current estimated length:', y_pred)
                    #print('current shape of all data: state: ', np.shape(s1_data), np.shape([s1_data_point]))
                    #print('current shape of all data: action: ', np.shape(a_data), np.shape([a_data_point]))
                    #print('current shape of all data: y: ', np.shape(y_data), np.shape(length))
                    s1_data = np.append(s1_data,[s1_data_point],0)
                    a_data = np.append(a_data,[a_data_point],0)
                    y_data = np.append(y_data,length)
                    #print('new shape of all data: state: ', np.shape(s1_data))
                    #print('new shape of all data: action: ', np.shape(a_data))
                    #print('new shape of all data: y: ', np.shape(y_data))


            ep_reward_list.append(episode_reward)
            print('EVALUATION RUN episode: ', ep + 1, 'reward:', episode_reward, 'true length ', length,
                  'final esimated length ', y_pred, 'current size of state data:', np.shape(s1_data), '\n')

        return s1_data, a_data, y_data



    def add_obs_to_window(self,tuple,obs,idx):
        # NOTE: tailored to fit state / action buffers, not really generic right now
        if ( idx < (len(tuple)-1) ) :
            tuple[idx] = obs
        else:
            #print('tuple shape',np.shape(tuple))
            #print('obs shape',np.shape([obs]))
            tuple = np.append(tuple[1:],[obs],0)
            #print('shape of new tuple:', np.shape(tuple))
        return tuple

    def test_policy(self,env,xml_path,lengths_list,model_names_list,render=False):
        ep_reward_list = []

        for ep in range(len(lengths_list)):
            length = lengths_list[ep]
            print("episode number", ep, "pendulum length:", length)
            env = self.change_model(env, xml_path, length)
            model_name = self.get_best_controller(length, lengths_list, model_names_list)
            self.my_load_actor(model_name)
            print('loaded model', model_name)

            episode_reward = 0
            state = env.reset()[0]
            idx = 0

            s1_data_point = np.zeros([self.hist_length, self.num_states])
            a_data_point = np.zeros([self.hist_length, self.num_actions])

            s1_data_point[idx] = state
            action = env.action_space.sample()
            a_data_point[idx] = action

            while True:
                if render: env.render()

                #just for kicks, what do we get?
                y_pred = self.sys_ID_network.predict([[s1_data_point], [a_data_point]], steps=1)
                #print('actual length ',length,'pridected length', y_pred)

                #s1_data_point[idx] = state
                s1_data_point = self.add_obs_to_window(s1_data_point,state,idx)
                action = self.get_action(state, noisy=False)
                a_data_point = self.add_obs_to_window(a_data_point,action,idx)
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                if done: break
                idx += 1

            ep_reward_list.append(episode_reward)
            print('EVALUATION RUN episode: ', ep+1, 'reward:', episode_reward, 'length ', length,
                  'final esimated length ', y_pred, '\n' )

    def my_save_actor(self, name):
        W_a = self.actor_model.get_weights()
        W_c = self.critic_model
        pickle.dump(W_a, open(name, 'wb'))
        #print('saving!')
        # print(W[0])

    def my_save_sysID(self,name):
        W = self.sys_ID_network.get_weights()
        pickle.dump(W, open(name, 'wb'))

    def my_load_actor(self, name):
        W_a = pickle.load(open(name, 'rb'))
        self.actor_model.set_weights(W_a)
        # print(W[0])

    def my_load_sysID(self,name):
        W = pickle.load(open(name, 'rb'))
        self.sys_ID_network.set_weights(W)



#######################################################################################################################
#model_names_list = ['InvPendv4_length6','InvPendv4_length5','InvPendv4_length4_b',
#                    'InvPendv4_length7','InvPendv4_length8', 'InvPendv4_length9']
lengths_list = [.6,.5,.4,.7,.8,.9]

environment = gym.make("InvertedPendulum-v4")
#name = 'models/InvPendv2_Jan06_b'


model_path = 'inverted_pendulum.xml'
if model_path.startswith("/"):
    xml_model_fullpath = model_path
else:
    xml_model_fullpath = os.path.join(os.path.dirname(__file__), ".venv/lib/site-packages/gym/envs/mujoco/assets", model_path)



my_agent = DDPG_agent(environment)
sysID_model_name = 'models/model_length_predictor_1'
AC_model_name = 'Parameterized_InvPendv2_Sep16_a'

# [s_data, a_data, y_data] = my_agent.generate_sysID_training_data(environment, model_names_list, lengths_list, xml_model_fullpath)
# [s_data, a_data, y_data] = my_agent.generate_sysID__bad_training_data(environment, xml_model_fullpath)
# [s_data, a_data, y_data] = my_agent.generate_ok_training_data_from_iffy_model
  #  (environment,xml_model_fullpath,lengths_list,model_names_list,'model_length_predictor_Aug2_c')
# pickle.dump([s_data, a_data, y_data], open('models/model_ID_imperfect_data_Aug3_a', 'wb'))
#[s_data, a_data, y_data] = pickle.load(open('models/model_ID_imperfect_data_Aug3_a', 'rb'))
#my_agent.train_sysID_network(s_data, a_data, y_data, sysID_model_name)

#my_agent.train_AC(environment,total_episodes=10000,render=False,
 #                 dynamic_model=False,xml_path=xml_model_fullpath,model_name=AC_model_name)

my_agent.my_load_actor(AC_model_name)
my_agent.my_load_sysID(sysID_model_name)

# for length in np.linspace(.3,1,8):
#     print('length:', length)
#     environment = my_agent.change_model(environment, xml_model_fullpath, length)
#     my_agent.evaluate(environment,length,episodes=1,render=True)
my_agent.online_sys_ID(environment,xml_model_fullpath,total_episodes=10000,render=False)

