import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
print("Eager:", tf.executing_eagerly())


class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        # return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        return tf.random.categorical(logits, 1)


class ActorCriticModel(keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()


    # Typically gets called only once (on first evaluation)
    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)


    def action_value(self, obs):
        # executes call() under the hood (on the first invocation)
        logits, value = self.predict(obs)
        # print("Logits:", logits)
        action = self.dist.predict(logits)
        # A simpler option would be:
        # action = tf.random.categorical(logits, 1)
        # However this wouldn't work if we turn of eager mode and use
        # static graph execution since we can't call random on graphs. only on values
        # return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
        return action, value

ac = ActorCriticModel(4)

ac.action_value(np.array([[1.2,2.3,3.6,4.7,5.8,3.9]],dtype=np.float32).reshape(3,2))

"""
Batch of size n

step  state  action value  reward done
0     s0     a0     v(0)   r(0)   d(0)
1     s1     a1     v(1)   r(1)   d(1)
...
n-1   s(n-1) a(n-1) v(n-1) r(n-1) d(n-1)

At step i we are in state s(i), with estimated value v(i) take action a(i),
receive reward r(i) and are done if d(i)

Then do 1 more step for v(n)

Construct cumulative reward R (size n)
R(i) = gamma R(i+1) + r(i)   if not done(i)  else  R(i) = r(i)
and R(n) = v(n)   (bootstrap)

Construct advantages A (size n):
A(i) = R(i) - v(i)

L(i) = [action(i), A(i)]

L will be used to train the logit output
R will be used to train the value output

losses = model.train on batch(state, [L, R])

"""

class SnakesA2C(Snakes):
    pass
