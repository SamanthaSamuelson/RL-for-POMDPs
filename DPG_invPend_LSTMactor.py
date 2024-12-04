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

# change 1: switch to out boring gaussian exploration noise:
        #  no change
# change 2: put everything into a DDPG_agent class, also: move learning from buffer object to agent object
        # no change
# change 3: switch to our buffer object
    # specifically - move sapmling step into buffer function
        # still seems to work
# change 4: switch to our create_actor_network function
    # specifically - no explicit initialization of network params
    # specifically - different post tanh scaling function
        # this makes it WORSE, but doesn't make it fail completely
# change 5: switch to our create_critic_network function
    # THIS DEFINITELY MADE IT FAIL
    # INTERMEDIATE RELU ACTIVATIONS SEEM IMPORTANT
# change 6: get_action vs policy functions - seem different...

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

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class NoiseObject: # cam make this fancy oblesteck noise later
    def __init__(self, mu, sigma):
        self.standard_deviation = sigma
        self.mean = mu #what if its not... bad for exploration, but good for robustness
        self.shape = 1

    def __call__(self):
        n = np.random.normal(self.mean, self.standard_deviation, self.shape)
        return n

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


    def get_training_batch(self): # should this be different for RNN?
        # move casting to tf here, to match source code better
        current_size = np.min([self.total_count, self.max_size])
        sample_indices = np.random.choice(current_size, self.batch_size, replace=True) # assumes you dont sample till buffer is full
        prev_state_batch = tf.convert_to_tensor(self.prev_state_buffer[sample_indices])
        prev_state_batch = tf.reshape(prev_state_batch, [self.batch_size, 1, self.state_size])
        action_batch = tf.convert_to_tensor(self.action_buffer[sample_indices])
        action_batch = tf.reshape(action_batch, [self.batch_size,1,1])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[sample_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[sample_indices])
        next_state_batch = tf.reshape(next_state_batch, [self.batch_size, 1, self.state_size])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[sample_indices])

        # order: state, action, reward, next state, done
        #buffers_tuple = [prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch]
        #return buffers_tuple
        return prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch


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
        #self.next_state_buffer = np.zeros([self.max_size, self.hist_length, self.state_size])
        #self.next_state_window = np.zeros([self.hist_length, self.state_size])
        self.action_buffer = np.zeros([self.max_size, self.hist_length, self.action_size])
        self.action_window = np.zeros([self.hist_length, self.action_size])
        #self.next_actions_buffer = np.zeros([self.max_size, self.hist_length, self.action_size])
        #self.next_actions_window = np.zeros([self.hist_length, self.action_size])

        # note: a history should have (ex) 11 states, 10 acions
        #self.trajectories_buffer = [0]*self.max_size
        #self.trajectory_window = [0]*(self.hist_length+1)

        self.reward_buffer = np.zeros([self.max_size, 1])
        #self.reward_window = np.zeros([self.hist_length, 1])
        self.dones_buffer = np.zeros([self.max_size, 1])
        #self.dones_window = np.zeros([self.hist_length, 1])


    def save_sample(self, prev_state, action, reward, state, done, step_number, save_frequency=10):
        # for storing sequences: put samples into window first, then save??
        # need to made sure sequences don't span episode breaks... take in step number as an input??
        # tuple order: state, action, reward, next state, done

        if (step_number == 0):
            # need to reset our windows!!
            self.reset_windows()
        #    self.trajectory_window[0:3] = [prev_state,action,state]
            self.state_window = np.vstack([self.state_window, prev_state])[-(self.hist_length+1):]
            #print('saved out starting state')
        else:
        #    self.trajectory_window = self.trajectory_window[2:] + [action, state]

        #self.state_window = np.vstack([self.state_window, prev_state])[-self.hist_length:]
            self.action_window = np.vstack([self.action_window, action])[-self.hist_length:]
            #print('action window:', self.action_window)
        #self.next_state_window = np.vstack([self.next_state_window, state])[-self.hist_length:]
        #self.next_actions_window = np.vstack([self.next_actions_window, next_action])[-self.hist_length:]
            self.state_window = np.vstack([self.state_window, state])[-(self.hist_length+1):]
            #print('state window', self.state_window)

        #self.reward_window = np.vstack([self.reward_window, reward])[-self.hist_length:]
        #self.dones_window = np.vstack([self.dones_window, done])[-self.hist_length:]

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
            #self.trajectories_buffer[self.storage_idx] = self.trajectory_window
            self.state_buffer[self.storage_idx] = self.state_window
            #print('state buffer:', self.state_buffer)
            self.action_buffer[self.storage_idx] = self.action_window
            #print('action buffer: ', self.action_buffer)
        #self.next_state_buffer[self.storage_idx] = self.next_state_window
        #self.next_actions_buffer[self.storage_idx] = self.next_actions_window

            self.total_count += 1


    def get_training_batch(self): # should this be different for RNN? actually it should be fine
        # move casting to tf here, to match source code better
        current_size = np.min([self.total_count, self.max_size])

        sample_indices = np.random.choice(current_size, self.batch_size, replace=True) # assumes you dont sample till buffer is full
        prev_state_batch = tf.convert_to_tensor(self.state_buffer[sample_indices,:-1,:])
        action_batch = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[sample_indices])
        next_state_batch = tf.convert_to_tensor(self.state_buffer[sample_indices,1:,:])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[sample_indices])
        #next_actions_batch = tf.convert_to_tensor(self.next_actions_buffer[sample_indices])
        #history_batch = tf.convert_to_tensor(list(itemgetter(*sample_indices)(self.trajectories_buffer)))

        # order: state, action, reward, next state, done
        #buffers_tuple = [prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch]
        #return buffers_tuple
        #return prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch, next_actions_batch
        return prev_state_batch, action_batch, next_state_batch, reward_batch, dones_batch

    def reset_windows(self):
        self.state_window = np.zeros([self.hist_length+1, self.state_size])
        #self.next_state_window = np.zeros([self.hist_length, self.state_size])
        self.action_window = np.zeros([self.hist_length, self.action_size])
        self.reward_window = np.zeros([self.hist_length, 1])
        self.dones_window = np.zeros([self.hist_length, 1])
        #self.next_actions_window = np.zeros([self.hist_length, self.action_size])




class DDPG_agent:
    def __init__(self, env):
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

        self.hist_length=10

        self.std_dev = 0.1
        self.noise_decay = 0.99
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        self.noise = NoiseObject(0, self.std_dev)

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
        self.buffer2 = ReplayBuffer_with_hist(self.buffer_size, self.batch_size, self.num_states, self.num_actions, self.hist_length)



    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    def create_actor_network(self):
        # input state, output action
        # note: if we use dropout or batch normalization use the training=TRUE (or false) flag when calling the model
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = Input(shape=(None, self.num_states))
        #state_input = Input(shape=(self.hist_length,self.num_states))  ## WANT TO BE ABLE TO SET STATE_SPACE SIZE AUTOMATICALLY, NEED TO FIX
        #state_input = Input(shape=self.state_space)

        i = state_input
        #i = Flatten()(i)
        i = LSTM(32)(i)

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
        def action_mapping_layer(x): return x * self.action_multiplier + self.action_bias
        i = Lambda(action_mapping_layer)(i)
        #i = Lambda(lambda x: x * self.action_multiplier + self.action_bias)(i)

        out = i
        model = Model(inputs=[state_input], outputs=out)

        def my_loss_fn(_, crit_of_pred):
            return -tf.reduce_mean(crit_of_pred)  # Note the `axis=-1`
        model.compile(loss=my_loss_fn, optimizer=self.actor_optimizer)
        # source doesn't compile, so for now we dont either
        return model
        # ok, mean squared error is really the wrong loss here - we dont use it anyway


    def create_critic_network(self):
        # input state, action, output value
        #state_input = Input(shape=(self.hist_length,self.num_states))
        state_input = Input(shape=(None, self.num_states))
        a = LSTM(32)(state_input)
        #state_input = Input(shape=self.state_space) ## WANT TO BE ABLE TO SET STATE_SPACE SIZE AUTOMATICALLY, NEED TO FIX

        #action_input = Input(shape=self.action_space)
        #action_input = Input(shape=(self.hist_length, self.num_actions))
        action_input = Input(shape=(None, self.num_actions))
        b = LSTM(16)(action_input)
        #b = Flatten()(action_input)

        #merged = Concatenate()([state_input, action_input])
        merged = Concatenate()([a,b])
        #merged = Flatten()(merged)
        #merged = LSTM(1)(merged)

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
        #state = np.reshape(state, (1, self.num_states))
        state = np.reshape(state, (1, 1, self.num_states))
        #print('new state of shape', np.shape(state))
        sample_action = self.actor_model(state) # model(state) vs model.predict(state) does not matter here
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
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        # Training and updating Actor & Critic networks.
        print('update function being called!')

        # # To train critic: we can probably do without gradient tape btw
        # next_actions = self.target_actor(state_batch)
        # expected_future_value = self.target_critic([next_state_batch, next_actions])
        # y = reward_batch + self.gamma * expected_future_value
        # self.critic_model.fit([state_batch, action_batch], y, verbose=0, steps_per_epoch=100)
        # # note: need to have COMPILE line uncommented in create_critic for this to work
        #
        # # can we train actor without gradient tape??
        # # note how the loss function goes ..
        # actions = self.actor_model(state_batch)
        # # print('got predicted actions - should these be the same as actions_batch?', actions)
        # critic_value_of_actions = self.critic_model([state_batch, actions])
        # # print('got predicted value of state action pairs', critic_value_of_actions)
        # self.actor_model.fit(state_batch, critic_value_of_actions, verbose=0, steps_per_epoch=100)
        # # print('omg did this work???')

        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            print('size of target actions:', tf.shape(target_actions))
            target_actions = tf.reshape(target_actions, [self.batch_size, 1, 1])
            print('reshaped target actions :', tf.shape(target_actions))
            print('size of fut. crit. ', tf.shape(self.target_critic([next_state_batch, target_actions])),
             'not dones has shape:', tf.shape(1 - dones_batch),  'size of reward batch', tf.shape(reward_batch))
            y = reward_batch + (1 - dones_batch)*self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            print('target critic values (y) has shape:', tf.shape(y))
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            print('predicted critic value has shape: ', tf.shape(critic_value))
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            print('critic loss value has shape: ', critic_loss)
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            print('suggested actions shape:', tf.shape(actions))
            actions = tf.reshape(actions, [self.batch_size,1,1])
            print('reshaped suggested actions shape:', tf.shape(actions))
            critic_value = self.critic_model([state_batch, actions], training=True)
            print('critic value of actions has shape: ', tf.shape(critic_value))
            #actions = self.actor_model(next_state_batch, training=True)
            #critic_value = self.critic_model([next_state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            #print('actor loss for updating: ', actor_loss, '\n')
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))


    def train(self, env, total_episodes=1000, render=False, dynamic_model=False, xml_path='', model_name='DDPG_model'):
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
                state, reward, done, info, _ = env.step(action)
                step_number+=1

                # order: state, action, reward, next state, done
                #self.buffer.save_sample((prev_state, action, reward, state, done))
                self.buffer2.save_sample(prev_state, action, reward, state, done, step_number, save_frequency=1)
                episodic_reward += reward

                #prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch = my_agent.buffer.get_training_batch()

                prev_state_batch2, action_batch2, next_state_batch2, reward_batch2, dones_batch2 = my_agent.buffer2.get_training_batch()
                #print('sizes of prev_state batches we got from buffer: next state batch', tf.shape(prev_state_batch).numpy()
                 #     , tf.shape(prev_state_batch2).numpy())
                print('sizes of action batches we got from buffer: next state batch',tf.shape(action_batch).numpy()
                      , tf.shape(action_batch2).numpy())
                #print('sizes of next_state batches we got from buffer: next state batch',tf.shape(next_state_batch).numpy()
                 #     , tf.shape(next_state_batch2).numpy())
                #print('sizes of reward batches we got from buffer: next state batch',
                 #     tf.shape(reward_batch).numpy(), tf.shape(reward_batch2).numpy())
                #print('sizes of dones batches we got from buffer: next state batch',
                 #     tf.shape(dones_batch).numpy(), tf.shape(dones_batch2).numpy())

                #print('calling update function \n')
                #self.update(prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch)
                self.update(prev_state_batch2, action_batch2, reward_batch2, next_state_batch2, dones_batch2)

                self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

                if done:
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
        #print('all state info:', old_state.time, old_state.qpos, old_state.qvel, old_state.act,
         #     old_state.udd_state)
        old_state = env.data


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
        env.reset()[0]

        print('resetting model to previous state', env.set_state(old_state.qpos, old_state.qvel))
        env.set_state(old_state.qpos, old_state.qvel)
        print('Done! \n')
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
                state, reward, done, info,_ = env.step(action)
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



name = 'InvPendv2_Aug07_LSTM_c'
length = .6

# maybe move num_training eps input around??
my_agent = DDPG_agent(environment)
environment = my_agent.change_model(environment, xml_model_fullpath, length)
my_agent.my_load(os.path.join("models", name))
#my_agent.train(environment,total_episodes=3000, dynamic_model=False, xml_path=xml_model_fullpath, model_name=name)
#my_agent.my_save(os.path.join("models", name))
my_agent.evaluate(environment, episodes=20)
#my_agent.my_load(name)
#my_agent.actor_model = models.load_model(os.path.join("models", name))
#my_agent.evaluate(environment, episodes=20)
