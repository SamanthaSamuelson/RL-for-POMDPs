Repository contains basic experiments in the performance of Q learning (specifically deep deterministic policy gradient) on the inverted pendulum problem as simulated by Mujoco.  '
The goal is to use simulations to provide some numerical answers to the question: How much does poor observatility cost in terms of the expected value of the problem? 

We include the possibility of three types of disturbance:
    1. Model inaccuracy or changing model: code allows for the possibility of changing the pendulum length, either while training or while testing the policy
        In this file it is possible to set the length of the inverted pendulum: standard length is 1, we vary lengths from .8 to 1.8.
    2. Observation noise: we can add Gaussian zero-mean noise to state observations while training and/or evaluation a policy
    3. Partial observability: The state space is four dimintional: linear cart position, angular pole position, linear cart velocity, angular pole velocity
        By truncating the state observation to 2 or three observations (out of 4) we make the problem partially obervable.  

File DPG_invPend_dynamic_betterReward.py trains and evaluates models under all three sources of disturbance.  
File DPG_invPend_DynamicModel_ModelIDtoControl performs in-the-loop model identification to control the inverted pendulum while the pendulum length changes.  
File DPG_invPend_LSTMactor adds an LSTM layer to both actor and critic networks in order to leverage the memroy properties of LSTM's in order to compensate for partial observatibliy, observation noise, or unknown model parameters

Ideally we will be able to match the value function learned for these problems settings to a theoretical lower bound on the relative sub-optimality of a value function learned in a partially observable setting for this specific inverted pendulum dynamical system.
