import gym
import tensorflow as tf
from tensorflow.keras import Model, losses
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input, Add, Activation, Lambda, Concatenate, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import numpy as np
import os
import pickle

tf.keras.backend.set_floatx('float64')

# Deep Deterministic Policy Gradient (actor-critic method) on the OpenAi pendulum swingup problem.  Policy is saved in 'DDPG_PendSwingUp_1'


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
        action_batch = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[sample_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[sample_indices])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[sample_indices])

        # order: state, action, reward, next state, done
        #buffers_tuple = [prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch]
        #return buffers_tuple
        return prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch







class DDPG_agent:
    def __init__(self,env):
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

        std_dev = 0.
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.noise = NoiseObject(0, std_dev)

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.actor_model = self.create_actor_network()
        self.critic_model = self.create_critic_network()

        self.target_actor = self.create_actor_network()
        self.target_critic = self.create_critic_network()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())


        self.total_episodes = 100
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

        self.buffer_size = 50000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.num_states, self.num_actions)



    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        #for (a, b) in zip(target_weights, weights):
        #    a.assign(b * tau + a * (1 - tau))
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def create_actor_network_orig(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=self.state_space)
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def create_actor_network(self):
        # input state, output action
        # note: if we use dropout or batch normalization use the training=TRUE (or false) flag when calling the model

        state_input = Input(shape=self.state_space)  ## WANT TO BE ABLE TO SET STATE_SPACE SIZE AUTOMATICALLY, NEED TO FIX
        i = state_input
        #i = Flatten()(i)
        i = Dense(128, activation="relu")(i)
        i = Dense(256, activation="relu")(i)
        #i = resdense(32)(i)
        #i = resdense(self.outputdims)(i)
        # map into (0,1)
        i = Dense(self.num_actions)(i)
        i = Activation('tanh')(i) # maps things to -1, 1
        # i = Activation('linear')(i)
        # map into action_space
        i = Lambda(lambda x: x * self.action_multiplier + self.action_bias)(i)

        out = i
        model = Model(inputs=[state_input], outputs=out)

        def my_loss_fn(_, crit_of_pred):
            return -tf.reduce_mean(crit_of_pred)  # Note the `axis=-1`
        #model.compile(loss=my_loss_fn, optimizer=self.actor_optimizer)
        # source doesn't compile, so for now we dont either
        return model
        # ok, mean squared error is really the wrong loss here - we dont use it anyway

    def create_critic_network_orig(self):
        # State as input
        state_input = layers.Input(shape=self.state_space)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=self.action_space)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def create_critic_network(self):
        # input state, action, output value
        state_input = Input(shape=self.state_space) ## WANT TO BE ABLE TO SET STATE_SPACE SIZE AUTOMATICALLY, NEED TO FIX
        h = Dense(16, activation="relu")(state_input)
        h = Dense(32, activation="relu")(h)

        action_input = Input(shape=self.action_space)
        j = Dense(32, activation="relu")(action_input)

        #merged = Add()([h,j])
        merged = Concatenate()([h,j])
        i = Dense(256, activation="relu")(merged)
        i = Dense(256, activation="relu")(i)
        i = Dense(1)(i)
        #i = Activation('relu')(i)
        out = i

        model = Model(inputs=[state_input, action_input], outputs=out)
        #model.compile(loss='mse', optimizer=self.critic_optimizer)

        return model


    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    def get_action(self, state):
        state = np.reshape(state, (1, self.num_states))
        sample_action = self.actor_model(state) # model(state) vs model.predict(state) does not matter here
        noisy_action = sample_action + self.noise()
        legal_action = self.clamper(noisy_action)

        return legal_action[0]
        # pay attention - this may be an environment thing

# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        # Train critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))


        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))



    def train(self, model_name):

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        max_reward = float('-inf')

        # Takes about 4 min to train
        for ep in range(self.total_episodes):

            prev_state = env.reset()
            action = env.action_space.sample()
            prev_state, reward, done, _, _ = env.step(action)

            episodic_reward = 0
            steps = 0

            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()

                #tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                #action = self.policy(tf_prev_state)
                action = self.get_action(prev_state)
                #print('returned actions are: tf version', action, 'and non-tf version', action2)
                # Recieve state and reward from environment.
                state, reward, done, info, _ = env.step(action)
                steps += 1

                # order: state, action, reward, next state, done
                self.buffer.save_sample((prev_state, action, reward, state, done))
                episodic_reward += reward

                # order: state, action, reward, next state, done
                prev_state_batch, action_batch, reward_batch, next_state_batch, dones_batch = my_agent.buffer.get_training_batch()
                self.update(prev_state_batch, action_batch, reward_batch, next_state_batch)

                #self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                #self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
                weights = []
                targets = self.target_critic.weights
                for i, weight in enumerate(self.critic_model.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_critic.set_weights(weights)
                weights = []
                targets = self.target_actor.weights
                for i, weight in enumerate(self.actor_model.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_actor.set_weights(weights)

                # End this episode when `done` is True
                #if done:
                if steps > 200:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
            if avg_reward >= max_reward:
                self.my_save( model_name)
                #self.my_save(os.path.join("models", model_name))
                max_reward = avg_reward
                print('saving run with average reward ', avg_reward)

    def demonstrate_policy(self, train_eps):

        ep_reward_list = []
        for ep in range(train_eps):

            prev_state = env.reset()
            action = env.action_space.sample()
            prev_state, reward, done, _, _ = env.step(action)

            episodic_reward = 0
            steps = 0

            while True:
                env.render()
                action = self.get_action(prev_state)
                state, reward, done, info, _ = env.step(action)
                steps += 1
                episodic_reward += reward
                if steps > 200:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)
            print("Episode  Reward ", episodic_reward)

        # Plotting graph
        # Episodes versus Avg. Rewards
        # plt.plot(avg_reward_list)
        # plt.xlabel("Episode")
        # plt.ylabel("Avg. Epsiodic Reward")
        # plt.show()

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



problem = "Pendulum-v1"
name = 'DDPG_PendSwingUp_1'

#env = gym.make(problem)
#my_agent = DDPG_agent(env)
#my_agent.train(name)
#env.close()

env = gym.make(problem, render_mode='human')
my_agent = DDPG_agent(env)
my_agent.my_load(name)
my_agent.demonstrate_policy(10)
