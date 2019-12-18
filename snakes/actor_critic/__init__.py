from snakes import Snakes, Rewards, TYPE_FLOAT, TYPE_BOOL, CHANNELS, CHANNEL_BODY, CHANNEL_HEAD, CHANNEL_APPLE
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.regularizers as kr
import tensorflow.keras.optimizers as ko
# import traceback

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


CONVOLUTION = False
DEBUG_INPUT = False

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
if is_interactive():
    print("Eager:", tf.executing_eagerly())


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # print("Call Probability", flush=True)
        # traceback.print_stack()
        # sample a random categorical action from given logits
        # Squeeze since each row of logits becomes a row with 1 action
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class ActorCriticModel(tf.keras.Model):
    def __init__(self, num_actions, height, width, data_format = "channels_last"):
        super().__init__('mlp_policy')

        self.dist = ProbabilityDistribution()

        # no tf.get_variable(), just simple Keras API
        if CONVOLUTION:
            self.model_conv(height, width, data_format)
        else:
            self.model_dense(height, width, data_format)
        self.value = kl.Dense(1, kernel_regularizer=kr.l2(0.0001), name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, kernel_regularizer=kr.l2(0.0001), name='policy_logits')


    def model_conv(self, height, width, data_format):
        self.conv1 = kl.Conv2D(filters = 3, kernel_size = 3,
                               activation='relu',
                               data_format = data_format,
                               input_shape=(height, width, 3))
        self.conv2 = kl.Conv2D(filters = 3, kernel_size = 3,
                               activation='relu',
                               data_format = data_format,
                               input_shape=(height, width, 3))
        self.pooling1 = kl.MaxPooling2D()
        self.pooling2 = kl.MaxPooling2D()
        self.flatten1 = kl.Flatten()
        self.flatten2 = kl.Flatten()


    def model_dense(self, height, width, data_format):
        if data_format == "channels_first":
            self.reshape = kl.Reshape((height*width*CHANNELS,),
                                      input_shape = (CHANNELS, height, width))
        elif data_format == "channels_last":
            self.reshape = kl.Reshape((height*width*CHANNELS,),
                                      input_shape = (height, width, CHANNELS))
        else:
            raise(ValueError("Unknown data_format:" + data_format))

        self.hidden1 = kl.Dense(128, activation='relu', kernel_regularizer=kr.l2(0.0001))
        self.hidden2 = kl.Dense(128, activation='relu', kernel_regularizer=kr.l2(0.0001))


    # Typically gets called only once (on first evaluation)
    @tf.function
    def call(self, inputs):
        # print("Call Model", flush=True)
        # traceback.print_stack()
        # inputs is a numpy array, convert to Tensor
        # x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = inputs

        if CONVOLUTION:
            # separate hidden layers from the same input tensor
            logs = self.conv1(x)
            logs = self.pooling1(logs)
            logs = self.flatten1(logs)

            vals = self.conv2(x)
            vals = self.pooling2(vals)
            vals = self.flatten2(vals)
        else:
            # logs = self.hidden1(x)
            # vals = self.hidden2(x)
            x = self.reshape(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            logs = x
            vals = x

        return self.logits(logs), self.value(vals)


    def action_value(self, obs):
        # print("Action_value")

        # I just want to call predict() here, not predict_on_batch()
        # But my current version of tensorflow (2.0) seems to leak memory
        # This may be https://github.com/keras-team/keras/issues/13118
        # And https://github.com/tensorflow/tensorflow/issues/33030
        # So this is possibly fixed in 2.1.0
        # Notice that predict_on_batch returns tensorflow arrays where
        # predict returns numpy arrays
        # executes call() under the hood (on the first invocation)
        logits, value = self.predict_on_batch(obs)
        # print("Logits:", logits)

        # Same comment applies: should really just be "predict"
        action = self.dist.predict_on_batch(logits)
        # A simpler option would be:
        # action = tf.random.categorical(logits, 1)
        # However this wouldn't work if we turn off eager mode and use static
        # graph execution since we can't call random on graphs, only on values
        # We can squeeze values since the last dense layer has a size of 1 unit
        return logits, action, np.squeeze(value, axis=-1)


if is_interactive():
    ac = ActorCriticModel(4, 4, 5)

    vals = np.arange(120,dtype=np.float32).reshape(-1,4,5,3)
    print(vals)

    print(ac.action_value(vals))

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
    def __init__(self, *args,
                 xy_head  = False,
                 xy_apple = False,
                 history_pit = True,
                 reward_file = None,
                 point_image = False,
                 # channels = 1,
                 channels = CHANNELS,
                 learning_rate = 0.1,
                 discount      = 0.9,
                 entropy_beta  = 0.0001,
                 **kwargs):

        if reward_file is None:
            self._rewards = Rewards.default()
        else:
            self._rewards = Rewards.parse_file(reward_file)

        if DEBUG_INPUT:
            channels = CHANNELS
            point_image = True
        if channels == 1:
            point_image = True
        super().__init__(*args,
                         xy_head = xy_head,
                         xy_apple = xy_apple,
                         history_pit = history_pit,
                         channels = channels,
                         point_image = point_image,
                         **kwargs)

        self._learning_rate = TYPE_FLOAT(learning_rate)
        self._discount = TYPE_FLOAT(discount)
        self._value_factor = 0.5
        self._entropy_beta = entropy_beta
        # self._eat_frame = np_empty(self.nr_snakes, TYPE_MOVES)
        self._model = ActorCriticModel(
            4,
            width = self.WIDTH,
            height = self.HEIGHT,
            data_format = "channels_last" if CONVOLUTION or channels > 1 else "channels_first")
        self._model.compile(
            # optimizer = ko.RMSprop(lr = self._learning_rate),
            optimizer = 'adam',
            # define separate losses for policy logits and value estimate
            loss = [self.loss_logits, self.loss_value])
        print("Eager model:", self._model.run_eagerly)
        print("Eager TF:", tf.executing_eagerly())


    @tf.function
    def loss_logits(self, actions_advantages, logits):
        # print("Logits_loss", type(actions_advantages), type(logits), flush=True)
        # traceback.print_stack()
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(actions_advantages, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on
        # call() from_logits argument ensures transformation into normalized
        # probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self._entropy_beta * entropy_loss


    @tf.function
    def loss_value(self, rewards, value):
        # print("Value_loss", type(rewards), type(value))
        # traceback.print_stack()
        # value loss is typically MSE between value estimates and returns
        return self._value_factor * kls.mean_squared_error(rewards, value)


    def log_constants(self, log_action):
        super().log_constants(log_action)
        log_action("ValueAlpha",  "%11.3f"  , self._value_factor)
        log_action("EntropyBeta", "%10.3f"  , self._entropy_beta)


    def log_frame(self, log_action, current):
        super().log_frame(log_action, current)
        if self._loss:
            log_action("Loss/Total",  "%11.3f", self._loss[0])
            log_action("Loss/Logits", "%10.3f", self._loss[1])
            log_action("Loss/Value",  "%11.3f", self._loss[2])


    def run_start(self, display):
        super().run_start(display)

        self._old_action = [None] * self.HISTORY
        self._old_values = [None] * self.HISTORY
        self._loss = None


    def move_select(self, move_result, display):
        debug = self.debug

        head  = self.head()
        apple = self.apple()

        h  = self.frame_history_now()
        h0 = self.frame_history_then()

        batch_size = 2048
        di, dj = divmod(self._debug_index, batch_size)
        dii = di * batch_size
        all_actions = []
        all_values  = []
        for i in range(0, self.nr_snakes, batch_size):
            i1 = min(i+batch_size, self.nr_snakes)
            r = range(i, i1)
            if self._channels == 1 or DEBUG_INPUT:
                input = [[self._field0[j],
                          self._point_image[apple[j]],
                          self._point_image[head[j]]] for j in r]
                # input = tf.convert_to_tensor(input, dtype=tf.float32)
                input = np.array(input, dtype=TYPE_FLOAT)
                if CONVOLUTION or DEBUG_INPUT:
                    # Move channels to the end
                    # Tensorflow CPU cannot handle channel first (GPU can)
                    input = np.rollaxis(input, 1,4)
                if DEBUG_INPUT:
                    assert np.array_equal(input, self._deep_field0[i:i1])
            else:
                input = self._deep_field0[i:i1]
            # print("NOW")
            # print(input[self._debug_index,:,:,CHANNEL_BODY])
            # print(input[self._debug_index,:,:,CHANNEL_HEAD])
            # print(input[self._debug_index,:,:,CHANNEL_APPLE])
            # The returned values are of type numpy.ndarray
            # Except logits which is of type tf.Tensor as long as
            # action_value uses predict_on_batch() instead of predict()
            logits, actions, values = self._model.action_value(input)
            del input
            # print(type(logits), type(actions), type(values))
            if debug and dii == i:
                p = np.exp(logits[dj])/sum(np.exp(logits[dj]))
                print("Logits", logits[dj], "p", p)
            del logits
            # print(actions, values)
            all_actions.append(actions)
            all_values.append(values)

        all_values = np.concatenate(all_values, axis=None)
        if h0 is not None:
            reward_moves = np.random.uniform(
                self._rewards.move - self._rewards.rand /2,
                self._rewards.move + self._rewards.rand /2)
            reward_moves = np.random.uniform(
                (reward_moves-self._rewards.rand/2) * self._history,
                (reward_moves+self._rewards.rand/2) * self._history,
                size=self.nr_snakes)
            rewards = self._history_gained * self._rewards.apple
            rewards += np.where(self._history_game0 == self._nr_games,
                                # Bootstrap from discounted best estimate
                                (all_values + reward_moves) * self._discount,
                                self._rewards.crash)
            # Calculate the advantage = current reward - predicted reward
            advantage = rewards - self._old_values[h0]

            old_head  = self._history_head0
            old_apple = self._history_apple0
            old_action = self._old_action[h0]

            if debug:
                print("Old Action = %d, Old Value = %f, New Value = %f" %
                      (old_action[di][dj],
                       self._old_values[h0][self._debug_index],
                       all_values[self._debug_index]))
                print("Reward = %f, Advantage = %f (Old Game = %d, New Game = %d)" %
                      (rewards[self._debug_index],
                       advantage[self._debug_index],
                       self._history_game0[self._debug_index],
                       self._nr_games[self._debug_index]))

            batch_size = 2048
            losses = [0, 0, 0]
            for i in range(0, self.nr_snakes, batch_size):
                i1 = min(i+batch_size, self.nr_snakes)
                r = range(i, i1)
                if self._channels == 1 or DEBUG_INPUT:
                    input = [[self._history_field0[j],
                              self._point_image[old_apple[j]],
                              self._point_image[old_head[j]]] for j in r]
                    # input = tf.convert_to_tensor(input, dtype=tf.float32)
                    input = np.array(input, dtype=TYPE_FLOAT)
                    if CONVOLUTION or DEBUG_INPUT:
                        # Move channels to the end
                        # Tensorflow CPU cannot handle channel first (GPU can)
                        input = np.rollaxis(input, 1,4)
                    if DEBUG_INPUT:
                        assert np.array_equal(input, self._deep_history_field0[i:i1])
                else:
                    input = self._deep_history_field0[i:i1]
                # print("THEN")
                # print(input[self._debug_index,:,:,CHANNEL_BODY])
                # print(input[self._debug_index,:,:,CHANNEL_HEAD])
                # print(input[self._debug_index,:,:,CHANNEL_APPLE])
                # print(self._model.metrics_names)
                # print(channels.shape)
                # print(old_action[i//batch_size].shape)
                # print(advantage.shape)
                # print(rewards.shape)
                # Combine action and advantage into shape (batch,2)
                # Each argument to train_on_batch must have the same number of
                # samples, so each shape should start with batch
                action_advantage = np.stack((old_action[i//batch_size], advantage[i:i1]), axis=1)
                loss = self._model.train_on_batch(
                    input,
                    [action_advantage, rewards[i:i1]])
                del input
                del action_advantage
                assert len(loss) == 3
                # 0: total loss (output_1_loss + output_2_loss)
                # 1: output_1_loss
                # 2: output_2_loss
                losses[0] += loss[0]
                losses[1] += loss[1]
                losses[2] += loss[2]

            self._loss = losses
            if debug:
                print("Loss", losses[0])

        self._old_action[h] = all_actions
        self._old_values[h] = all_values
        if debug:
            print("New Value = %f, New Action = %d" % (all_values[self._debug_index], all_actions[di][dj]))
        pos = head + self.DIRECTIONS[np.concatenate(all_actions, axis=None)]
        return pos
