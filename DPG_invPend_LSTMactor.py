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

tf.keras.backend.set_floatx('float64')

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


class ReplayBuffer_with_hist:
    def __init__(self, buffer_size, batch_size, state_size, action_size, history_length):
        self.state_size = state_size
        self.action_size = action_size
        self.hist_length = history_length
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.storage_idx = 0
        self.total_count = 0

        self.state_buffer = np.zeros([self.max_size, self.hist_length+1, self.state_size])
        self.state_window = np.zeros([self.hist_length+1, self.state_size])
        self.action_buffer = np.zeros([self.max_size, self.hist_length, self.action_size])
        self.action_window = np.zeros([self.hist_length, self.action_size])
        # note: a history should have (ex) 11 states, 10 actions

        self.reward_buffer = np.zeros([self.max_size, 1])
        self.dones_buffer = np.zeros([self.max_size, 1])


    def save_sample(self, prev_state, action, reward, state, done, step_number, save_frequency=10):
        # for storing sequences: put samples into window first, then save??
        # need to made sure sequences don't span episode breaks... take in step number as an input??
        # tuple order: state, action, reward, next state, done

        if (step_number == 0):
            # need to reset our windows!!
            self.reset_windows()
            self.state_window = np.vstack([self.state_window, prev_state])[-(self.hist_length+1):]
            #print('saved out starting state')
        else:
            self.action_window = np.vstack([self.action_window, action])[-self.hist_length:]
            #print('action window:', self.action_window)
            self.state_window = np.vstack([self.state_window, state])[-(self.hist_length+1):]
            #print('state window', self.state_window)

        # SHOULD WE ONLY USE SEQUENCES ONCE THEY'RE FULL?? - yes, and this will also guarantee no episode wraparounds
            # but for early examples they may not fill up??
            # maybe store window every ... 10 steps or so? someting like that
        self.storage_idx = self.total_count % self.max_size
        #print('storage idx: ', self.storage_idx)

        # order: state, action, reward, next state, done

        if done | (step_number % save_frequency == 0):
            # if we failed, store the window / sample NOW in the buffer
            # or if _ steps have gone by since we last saved
            #print('saving sequence into buffer \n')
            self.reward_buffer[self.storage_idx] = reward
            #print('reward buffer shape', np.shape(self.reward_buffer))
            self.dones_buffer[self.storage_idx] = done
            self.state_buffer[self.storage_idx] = self.state_window
            #print('state buffer:', self.state_buffer)
            self.action_buffer[self.storage_idx] = self.action_window
            #print('action buffer: ', self.action_buffer)

            self.total_count += 1


    def get_training_batch(self): # should this be different for RNN? actually it should be fine
        current_size = np.min([self.total_count, self.max_size])

        sample_indices = np.random.choice(current_size, self.batch_size, replace=True) # assumes you dont sample till buffer is full
        prev_state_batch = tf.convert_to_tensor(self.state_buffer[sample_indices,:-1,:])
        action_batch = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[sample_indices])
        next_state_batch = tf.convert_to_tensor(self.state_buffer[sample_indices,1:,:])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[sample_indices])

        # order: state, action, reward, next state, done
        return prev_state_batch, action_batch, next_state_batch, reward_batch, dones_batch

    def reset_windows(self):
        self.state_window = np.zeros([self.hist_length+1, self.state_size])
        self.action_window = np.zeros([self.hist_length, self.action_size])
        self.reward_window = np.zeros([self.hist_length, 1])
        self.dones_window = np.zeros([self.hist_length, 1])




class DDPG_agent:
    def __init__(self, env):
        self.num_states = env.observation_space.shape[0] # gives integer
        self.state_space = (self.num_states,)
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

        self.hist_length=10

        self.std_dev = 0.1
        self.noise_decay = 0.99
        self.obs_std_dev = 0.05

        # Learning rate for actor-critic models
        # Discount factor for future rewards
        self.gamma = 0.9995
        # Used to update target networks
        self.tau = 0.002
        self.critic_lr = 0.001
        self.actor_lr = 0.001
        self.critic_optimizer = Adam(self.critic_lr)
        self.actor_optimizer = Adam(self.actor_lr)

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
        self.buffer2 = ReplayBuffer_with_hist(self.buffer_size, self.batch_size, self.num_states, self.num_actions, self.hist_length)



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

        state_input = Input(shape=(None, self.num_states))
        i = state_input
        i = LSTM(32)(i)
        i = resdense(64)(i)
        i = resdense(64)(i)
        i = resdense(32)(i)

        # map into (0,1)
        i = Dense(self.num_actions, kernel_initializer=last_init)(i)
        i = Activation('tanh')(i) # maps things to -1, 1
        # map into action_space
        def action_mapping_layer(x): return x * self.action_multiplier + self.action_bias
        i = Lambda(action_mapping_layer)(i)

        out = i
        model = Model(inputs=[state_input], outputs=out)

        def my_loss_fn(_, crit_of_pred):
            return -tf.reduce_mean(crit_of_pred)  # Note the `axis=-1`
        model.compile(loss=my_loss_fn, optimizer=self.actor_optimizer)
        return model


    def create_critic_network(self):
        # input state, action, output value
        #state_input = Input(shape=(self.hist_length,self.num_states))
        state_input = Input(shape=(None, self.num_states))
        a = LSTM(32)(state_input)
        action_input = Input(shape=(None, self.num_actions))
        b = LSTM(16)(action_input)

        merged = Concatenate()([a,b])

        i = resdense(64)(merged)
        i = resdense(64)(i)
        i = resdense(32)(i)
        i = Dense(1)(i)
        out = i

        model = Model(inputs=[state_input, action_input], outputs=out)
        model.compile(loss='mse', optimizer=self.critic_optimizer)

        return model



    def get_action(self, state, noisy=True):
        state = np.reshape(state, (1, 1, self.num_states))
        #print('new state of shape', np.shape(state))
        sample_action = self.actor_model(state) # model(state) vs model.predict(state) does not matter here
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
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        # Training and updating Actor & Critic networks.
        # print('update function being called!')

        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            #print('size of target actions:', tf.shape(target_actions))
            target_actions = tf.reshape(target_actions, [self.batch_size, 1, 1])
            #print('reshaped target actions :', tf.shape(target_actions))
            #print('size of fut. crit. ', tf.shape(self.target_critic([next_state_batch, target_actions])),
            # 'not dones has shape:', tf.shape(1 - dones_batch),  'size of reward batch', tf.shape(reward_batch))
            y = reward_batch + (1 - dones_batch)*self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            #print('target critic values (y) has shape:', tf.shape(y))
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            #print('predicted critic value has shape: ', tf.shape(critic_value))
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            #print('critic loss value has shape: ', critic_loss)
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            #print('suggested actions shape:', tf.shape(actions))
            actions = tf.reshape(actions, [self.batch_size,1,1])
            #print('reshaped suggested actions shape:', tf.shape(actions))
            critic_value = self.critic_model([state_batch, actions], training=True)
            #print('critic value of actions has shape: ', tf.shape(critic_value))
            actor_loss = -tf.math.reduce_mean(critic_value)
            #print('actor loss for updating: ', actor_loss, '\n')
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))


    def train(self, env, total_episodes=1000, render=False, dynamic_model=False, xml_path=''):
        total_steps = 0
        change_idx = np.random.choice(range(50000,60000),1)
        print('model should change at ', change_idx)

        ep_reward_list = []
        avg_reward_list = []
        max_reward = -np.inf

        for ep in range(total_episodes):

            prev_state = env.reset()[0]
            episodic_reward = 0
            step_number = 0
            self.buffer2.save_sample(prev_state, 0, 0, 0, 0, step_number)

            while True:
                if render: env.render()

                action = self.get_action(prev_state)
                state, reward, done, _, _ = env.step(action)
                ## Here add option for observation noise or truncated observations
                state = state[0:self.num_states] + np.random.normal(0, self.obs_std_dev, self.num_states)
                ## Introduce more sophisticated reward function, which gives larger reward the closer the pendulum is to vertical
                reward = math.pi/2-state[1]
                step_number += 1

                # order: state, action, reward, next state, done
                self.buffer2.save_sample(prev_state, action, reward, state, done, step_number, save_frequency=1)
                episodic_reward += reward

                prev_state_batch2, action_batch2, next_state_batch2, reward_batch2, dones_batch2 = my_agent.buffer2.get_training_batch()
                #print('sizes of prev_state batches we got from buffer: next state batch', tf.shape(prev_state_batch).numpy()
                 #     , tf.shape(prev_state_batch2).numpy())
                #print('sizes of action batches we got from buffer: next state batch',tf.shape(action_batch).numpy()
                      , tf.shape(action_batch2).numpy())
                #print('sizes of next_state batches we got from buffer: next state batch',tf.shape(next_state_batch).numpy()
                 #     , tf.shape(next_state_batch2).numpy())
                #print('sizes of reward batches we got from buffer: next state batch',
                 #     tf.shape(reward_batch).numpy(), tf.shape(reward_batch2).numpy())
                #print('sizes of dones batches we got from buffer: next state batch',
                 #     tf.shape(dones_batch).numpy(), tf.shape(dones_batch2).numpy())

                #print('calling update function \n')
                self.update(prev_state_batch2, action_batch2, reward_batch2, next_state_batch2, dones_batch2)

                self.update_target(self.target_actor, self.actor_model)
                self.update_target(self.target_critic, self.critic_model)

                if done or step_number > 1000:
                    break

                prev_state = state
                self.std_dev = self.std_dev*self.noise_decay

                total_steps += 1
                if (total_steps == change_idx) and (dynamic_model == True):
                    print('total steps are:', total_steps, 'changing model!')
                    total_steps = 0
                    change_idx = np.random.choice(range(1000,5000), 1)
                    env = self.change_model(env, xml_path, .6)
                    print('new change index:', change_idx, 'total steps at:', total_steps)

            ep_reward_list.append(episodic_reward)
            #if (ep % 10 == 0): print('Episode', ep, 'total reward is:', episodic_reward)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-10:])
            avg_reward_list.append(avg_reward)
            #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            if (ep % 10 == 0):
                print('Episode', ep, 'average reward is:', avg_reward)
            if avg_reward >= max_reward:
                #self.actor_model.save(os.path.join("models", model_name))
                self.my_save(os.path.join("models", model_name))
                max_reward = avg_reward
                print('saving run with average reward ', avg_reward)

            if (ep % 100 == 0): self.evaluate(env, episodes = 1)

    def change_model(self, env, xml_path, new_pend_length):
        # print('CHANGING PARAMETERS NOW')
        et = xml.etree.ElementTree.parse(xml_path)
        #print('xml object', et)
        root = et.getroot()
        #for child in root:
        #    print(child.tag, child.attrib)

        pendulum_geom_lengths = root[4][1][2][1].get('fromto')
        pendulum_geom_density = root[4][1][2][1].get('density')
        #print('current pole length', pendulum_geom_lengths)
        #print('current pole density', pendulum_geom_density)

        old_state = env.data

        pendulum_geom_lengths = '0 0 0 0.001 0 {}'.format(new_pend_length)
        #print('new pendulum (half) length .6')
        new_pend_density = 1000
        pendulum_geom_density = '{}'.format(new_pend_density)
        #print('new pendulum density', pendulum_geom_density)
        root[4][1][2][1].set('fromto', pendulum_geom_lengths)
        root[4][1][2][1].set('density', pendulum_geom_density)
        #print('trying to access pole length', root[4][1][2][1].attrib)
        #print('trying to access pole density', root[4][1][2][1].attrib)

        et.write(xml_model_fullpath)
        #print('xml file has been written')

        #print('re-make env - do we need to return this??')
        env = gym.make("InvertedPendulum-v4")
        env.reset()[0]

        #print('resetting model to previous state', env.set_state(old_state.qpos, old_state.qvel))
        env.set_state(old_state.qpos, old_state.qvel)
        #print('Done! \n')
        return env

    def evaluate(self, env, episodes=10, render=False):
        ep_reward_list = []
        for ep in range(episodes):
            episode_reward = 0
            steps = 0
            state = env.reset()[0]
            while True:
                if render: env.render()

                action = self.get_action(state, noisy=False)
                state, reward, done, _, _ = env.step(action)
                ## Here add option for observation noise or truncated observations
                state = state[0:self.num_states] + np.random.normal(0, self.obs_std_dev, self.num_states)
                ## Introduce more sophisticated reward function, which gives larger reward the closer the pendulum is to vertical
                reward = math.pi/2-state[1]
                episode_reward += reward
                steps += 1
                if done or steps > 1000:
                    break

            ep_reward_list.append(episode_reward)
            print('EVALUATION RUN episode: ', ep, 'reward:', episode_reward)

    def my_save(self, name):
        W_a = self.actor_model.get_weights()
        W_c = self.critic_model
        pickle.dump(W_a, open(name, 'wb'))
        #print('saving!')
        # print(W[0])
        models.save

    def my_load(self, name):
        W_a = pickle.load(open(name, 'rb'))
        self.actor_model.set_weights(W_a)






environment = gym.make("InvertedPendulum-v4")

model_path = 'inverted_pendulum.xml'
if model_path.startswith("/"):
    xml_model_fullpath = model_path
else:
    xml_model_fullpath = os.path.join(os.path.dirname(__file__), ".venv/lib/site-packages/gym/envs/mujoco/assets", model_path)



name = 'InvPendv4_b'
length = .6

my_agent = DDPG_agent(environment)
environment = my_agent.change_model(environment, xml_model_fullpath, length)
my_agent.my_load(os.path.join("models", name))
#my_agent.train(environment,total_episodes=3000, dynamic_model=False, xml_path=xml_model_fullpath)
#my_agent.my_save(os.path.join("models", name))
my_agent.evaluate(environment, episodes=20)
#my_agent.my_load(name)
#my_agent.actor_model = models.load_model(os.path.join("models", name))
#my_agent.evaluate(environment, episodes=20)
