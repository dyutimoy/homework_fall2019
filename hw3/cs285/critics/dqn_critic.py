from .base_critic import BaseCritic
import tensorflow as tf
from cs285.infrastructure.dqn_utils import minimize_and_clip, huber_loss,PiecewiseSchedule



class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']
        self.num_timesteps=hparams['num_timesteps']
        self.optimizer_spec = optimizer_spec
        self.q_func=hparams['q_func']
        self.q_t_values_model=None
        self.target_q_func_model=None
        self.clip_val=hparams['grad_norm_clipping']
        #self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-04)
        #self.define_placeholders()
        #self._build(hparams['q_func'])

    def build_model(self,obs):
        input_shape=tf.shape(obs)[1:]
        #tf.print(input_shape)
        
        if self.q_t_values_model is None:
                self.q_t_values_model= self.q_func(input_shape, self.ac_dim)
        if self.target_q_func_model is None:
                self.target_q_func_model=self.q_func(input_shape, self.ac_dim)  

    def save(self,path):
        self.q_t_values_model.save_weights(path)
    
    def build_q_t_val(self,obs):
        self.obs_t_ph=tf.convert_to_tensor(obs)
        if self.q_t_values_model is None:
          self.q_t_values_model= self.q_func(self.obs_t_ph, self.ac_dim)
        self.q_t_values =self.q_t_values_model(self.obs_t_ph)

    
    def train_one_step(self,ob_no, ac_na, re_n, next_ob_no, terminal_n,lr):

        #####################
        self.obs_t_ph=tf.convert_to_tensor(ob_no)
        self.act_t_ph=tf.convert_to_tensor(ac_na)
        self.rew_t_ph=tf.convert_to_tensor(re_n)
        self.obs_tp1_ph=tf.convert_to_tensor(next_ob_no)
        self.done_mask_ph=tf.convert_to_tensor(terminal_n)
        # q values, created with the placeholder that holds CURRENT obs (i.e., t)
        
                
        with tf.GradientTape() as tape:

            tape.watch(self.obs_t_ph)
            
                #pass
            self.q_t_values =self.q_t_values_model(self.obs_t_ph)
            #self.q_t_values=tf.Variable(self.q_t_values)
            #tf.print(self.q_t_values)
            self.q_t = tf.reduce_sum(self.q_t_values * tf.one_hot(self.act_t_ph, self.ac_dim), axis=1)
            #self.q_t=tf.Variable(self.q_t)
            #tf.print(self.q_t)
            #####################

            # target q values, created with the placeholder that holds NEXT obs (i.e., t+1)
     
            
            q_tp1_values = self.target_q_func_model(self.obs_tp1_ph)
            if self.double_q:
                # You must fill this part for Q2 of the Q-learning potion of the homework.
                # In double Q-learning, the best action is selected using the Q-network that
                # is being updated, but the Q-value for this action is obtained from the
                # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
                pass
            else:
                # q values of the next timestep
                q_tp1 = tf.reduce_max(q_tp1_values, axis=1)

            #####################

            # TODO calculate the targets for the Bellman error
            # HINT1: as you saw in lecture, this would be:
                #currentReward + self.gamma * qValuesOfNextTimestep * (1 - self.done_mask_ph)
            # HINT2: see above, where q_tp1 is defined as the q values of the next timestep
            # HINT3: see the defined placeholders and look for the one that holds current rewards
            target_q_t = self.rew_t_ph+self.gamma*(q_tp1)*(1-self.done_mask_ph)
            target_q_t = tf.stop_gradient(target_q_t)

            #####################

            # TODO compute the Bellman error (i.e. TD error between q_t and target_q_t)
            # Note that this scalar-valued tensor later gets passed into the optimizer, to be minimized
            # HINT: use reduce mean of huber_loss (from infrastructure/dqn_utils.py) instead of squared error
            self.total_error= tf.reduce_mean(huber_loss(self.q_t-target_q_t))
            #tf.print(self.total_error)
            #print(self.total_error)
            #####################

            # TODO these variables should all of the 
            # variables of the Q-function network and target network, respectively
            # HINT1: see the "scope" under which the variables were constructed in the lines at the top of this function
            # HINT2: use tf.get_collection to look for all variables under a certain scope
            #q_func_vars =self.q_t_values_model.trainable_weights
            #target_q_func_vars = TODO

        #####################
        #q_weights=self.q_t_values_model.trainable_weights
        gradients=tape.gradient(self.total_error,self.q_t_values_model.trainable_variables)
        #tf.print("trainable_weights",q_weights)
        # train_fn will be called in order to train the critic (by minimizing the TD error)
        self.learning_rate = lr
        #minimize_and_clip(optimizer, gradients, clip_val=self.grad_norm_clipping)
        processed_grads,_=tf.clip_by_global_norm(gradients, self.clip_val)
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)      
        optimizer.apply_gradients(zip(processed_grads,self.q_t_values_model.trainable_variables))

        # update_target_fn will be called periodically to copy Q network to target Q network

    def update_model(self):    
        
        self.q_t_values_model.set_weights(self.target_q_func_model.get_weights())
    """
    def define_placeholders(self):
        # set up placeholders
        # placeholder for current observation (or state)
        lander = self.env_name == 'LunarLander-v2'

        self.obs_t_ph = tf.Variable(
            tf.float32 if lander else tf.uint8, [None] + list(self.input_shape))
        # placeholder for current action
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        # placeholder for current reward
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(self.input_shape))
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph = tf.placeholder(tf.float32, [None])
    """
    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        raise NotImplementedError

