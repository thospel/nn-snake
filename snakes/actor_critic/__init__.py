from snakes import Snakes, Rewards, TYPE_FLOAT, TYPE_BOOL, CHANNELS, CHANNEL_BODY, CHANNEL_HEAD, CHANNEL_APPLE, CHANNEL_TAIL
import numpy as np
import math
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
DEBUG_INPUT_PRINT = False
DEBUG_TRAIN = False
BUGGY = False
NORMALIZE = False

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
    def __init__(self, num_actions, height, width):
        super().__init__('mlp_policy')

        self.dist = ProbabilityDistribution()

        # no tf.get_variable(), just simple Keras API
        if CONVOLUTION:
            self.model_conv(height, width)
        else:
            self.model_dense(height, width)
        self.value = kl.Dense(1, kernel_regularizer=kr.l2(0.0001), name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, kernel_regularizer=kr.l2(0.0001), name='policy_logits')


    def model_conv(self, height, width):
        self.conv1 = kl.Conv2D(filters = 8, kernel_size = 3,
                               padding = "same",
                               activation='relu',
                               kernel_regularizer=kr.l2(0.0001),
                               data_format = "channels_last",
                               input_shape=(height, width, 3))
        self.conv2 = kl.Conv2D(filters = 3, kernel_size = 3,
                               padding = "same",
                               activation='relu',
                               kernel_regularizer=kr.l2(0.0001),
                               data_format = "channels_last")
        # self.pooling1 = kl.MaxPooling2D()
        # self.pooling2 = kl.MaxPooling2D()
        self.flatten = kl.Flatten()
        # self.flatten2 = kl.Flatten()


    def model_dense(self, height, width):
        self.reshape = kl.Reshape((height*width*CHANNELS,),
                                  input_shape = (height, width, CHANNELS))
        self.hidden1 = kl.Dense(128,
                                activation='relu',
                                kernel_regularizer=kr.l2(0.0001))
        self.hidden2 = kl.Dense(128,
                                activation='relu',
                                kernel_regularizer=kr.l2(0.0001))


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
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.flatten(x)
            logs = x
            vals = x
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
                 # history_channels = 1,
                 channels = CHANNELS,
                 history_channels = CHANNELS,
                 single        = False,
                 learning_rate = 0.1,
                 discount      = 0.9,
                 entropy_beta  = 0.01,
                 value_weight  = 0.5,
                 switch_period = 40,
                 **kwargs):

        if reward_file is None:
            self._rewards = Rewards.default()
        else:
            self._rewards = Rewards.parse_file(reward_file)

        if DEBUG_INPUT:
            channels = CHANNELS
            history_channels = CHANNELS
            point_image = True
        if channels == 1 or history_channels == 1:
            point_image = True
        super().__init__(*args,
                         xy_head = xy_head,
                         xy_apple = xy_apple,
                         history_pit = history_pit,
                         channels = channels,
                         history_channels = history_channels,
                         point_image = point_image,
                         **kwargs)

        # We subtract crash from the reward when a game ends.
        # Compensate for that if we won
        self._win_bonus += math.ceil(-self._rewards.crash / self._rewards.apple)

        self._single = single
        self._learning_rate = TYPE_FLOAT(learning_rate)
        if not self._single:
            self._learning_rate /= self.nr_snakes
        self._discount = TYPE_FLOAT(discount)
        self._value_weight = value_weight
        self._entropy_beta = entropy_beta
        self._switch_period = switch_period
        # self._eat_frame = np_empty(self.nr_snakes, TYPE_MOVES)
        self._model_run = ActorCriticModel(
            4,
            width  = self.WIDTH,
            height = self.HEIGHT)
        self._model_run.compile(
            # optimizer = ko.RMSprop(lr = self._learning_rate),
            optimizer = ko.Adam(lr = self._learning_rate),
            # define separate losses for policy logits and value estimate
            loss = [self.loss_logits, self.loss_value])
        # self._model_train = ActorCriticModel(
        #    4,
        #    width = self.WIDTH,
        #    height = self.HEIGHT)
        #self._model_train.compile(
        #    # optimizer = ko.RMSprop(lr = self._learning_rate),
        #    optimizer = 'adam',
        #    # define separate losses for policy logits and value estimate
        #    loss = [self.loss_logits, self.loss_value])
        self._model_train = self._model_run
        print("Eager model:", self._model_run.run_eagerly)
        print("Eager TF:", tf.executing_eagerly())


    @tf.function
    def loss_logits(self, actions_advantages, logits):
        # print("Logits_loss", actions_advantages, logits)
        # traceback.print_stack()
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(actions_advantages, 2, axis=-1)
        # print("Actions", actions, "Advantages", advantages)
        # print("Logits", logits)
        # sparse categorical CE loss obj that supports sample_weight arg on
        # call() from_logits argument ensures transformation into normalized
        # probabilities
        # SparseCategoricalCrossentropy(a,b) = CategoricalCrossentropy(one_hot(a), b) / num_actions
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        if BUGGY:
            entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        else:
            # print("Policy loss", policy_loss)
            # entropy loss can be calculated via CE over itself
            probabilities = tf.nn.softmax(logits)
            entropy_loss = kls.categorical_crossentropy(probabilities, probabilities)
        # print("Entropy loss", entropy_loss)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self._entropy_beta * entropy_loss


    @tf.function
    def loss_value(self, rewards, value):
        # print("Value_loss", rewards, value)
        # traceback.print_stack()
        # value loss is typically MSE between rewards and value estimates
        return self._value_weight * kls.mean_squared_error(rewards, value)


    def log_constants(self, log_action):
        super().log_constants(log_action)

        lr = self._learning_rate
        if not self._single:
            lr *= self.nr_snakes

        log_action("Learning Rate", "%8.3f",  lr)
        log_action("ValueAlpha",  "%11.3f"  , self._value_weight)
        log_action("EntropyBeta", "%10.3f"  , self._entropy_beta)
        log_action("Reward apple",  "%9.3f",  self._rewards.apple)
        log_action("Reward crash",  "%9.3f",  self._rewards.crash)
        log_action("Reward move",   "%10.3f", self._rewards.move)
        log_action("Reward rand",   "%10.3f", self._rewards.rand)
        log_action("Reward init",   "%10.3f", self._rewards.initial)


    def log_frame(self, log_action, current):
        super().log_frame(log_action, current)
        if self._loss:
            log_action("Loss/Total",  "%11.3f", self._loss[0])
            log_action("Loss/Logits", "%10.3f", self._loss[1])
            log_action("Loss/Value",  "%11.3f", self._loss[2])

    def input_equal(self, offset, constructed, deep):
        if np.array_equal(constructed, deep):
            return
        if constructed.shape != deep.shape:
            raise(AssertionError("Constructed and deep input shapes differ"))
        for i in range(deep.shape[0]):
            if np.array_equal(constructed[i], deep[i]):
                continue
            print("CONSTRUCTED")
            print(constructed[i,:,:,CHANNEL_BODY].astype(np.int8))
            print(constructed[i,:,:,CHANNEL_HEAD].astype(np.int8))
            print(constructed[i,:,:,CHANNEL_TAIL].astype(np.int8))
            print(constructed[i,:,:,CHANNEL_APPLE].astype(np.int8))
            print("DEEP")
            print(deep[i,:,:,CHANNEL_BODY].astype(np.int8))
            print(deep[i,:,:,CHANNEL_HEAD].astype(np.int8))
            print(deep[i,:,:,CHANNEL_TAIL].astype(np.int8))
            print(deep[i,:,:,CHANNEL_APPLE].astype(np.int8))
            raise(AssertionError("Constructed and deep input differ"))
        raise(AssertionError("Constructed and deep input differ, but I can't find the difference"))


    def run_start(self, display):
        super().run_start(display)

        self._old_action = [None] * self.HISTORY
        self._old_values = [None] * self.HISTORY
        self._loss = None


    def fit(self, old_action, advantage, rewards):
        debug = self.debug
        batching = self._history_channels == 1 or DEBUG_INPUT

        if batching:
            old_apple = self._history_apple0
            old_head  = self._history_head0
            frame0 = self.frame_then()
            old_tail = self._snake_body[self._all_snakes, (frame0 - self._history_score) & self.MASK]
            batch_size = 2048
        else:
            batch_size = self.nr_snakes

        di, dj = divmod(self._debug_index, batch_size)
        dii = di * batch_size

        action_advantage = np.stack((old_action, advantage), axis=1)

        losses = [0, 0, 0]
        for i in range(0, self.nr_snakes, batch_size):
            i1 = min(i+batch_size, self.nr_snakes)
            if batching:
                input = [[self._history_field0[j],
                          self._point_image[old_apple[j]],
                          self._point_image[old_head[j]],
                          self._point_image[old_tail[j]]
                ] for j in range(i, i1)]
                # input = tf.convert_to_tensor(input, dtype=tf.float32)
                input = np.array(input, dtype=TYPE_FLOAT)
                # Tensorflow CPU cannot handle channel first (GPU can)
                # Move channels to the end (this doesn't copy)
                # (of course it will slow down the copy to tensor)
                input = np.rollaxis(input, 1,4)
                if DEBUG_INPUT:
                    self.input_equal(i, input, self._deep_history_field0[i:i1])
            else:
                # input = self._deep_history_field0[i:i1]
                input = self._deep_history_field0
            if DEBUG_INPUT_PRINT and dii == i:
                print("THEN")
                print(input[dj,:,:,CHANNEL_BODY].astype(np.int8))
                print(input[dj,:,:,CHANNEL_HEAD].astype(np.int8))
                print(input[dj,:,:,CHANNEL_TAIL].astype(np.int8))
                print(input[dj,:,:,CHANNEL_APPLE].astype(np.int8))
            if debug and DEBUG_TRAIN and dii == i:
                logits, _, values = self._model_train.action_value(input[dj:dj+1])
                p = np.exp(logits[dj])/sum(np.exp(logits[dj]))
                print("Value  Before Train", values[dj])
                print("Logits Before Train", logits[dj], "p", p)

            # print(self._model_train.metrics_names)
            # print(channels.shape)
            # print(old_action[i//batch_size].shape)
            # print(advantage.shape)
            # print(rewards.shape)
            # Combine action and advantage into shape (batch,2)
            # Each argument to train_on_batch must have the same number of
            # samples, so each shape should start with batch
            if batching:
                loss = self._model_train.train_on_batch(
                    input,
                    [action_advantage[i:i1], rewards[i:i1]])
            elif False:
                loss = self._model_train.train_on_batch(
                    input,
                    [action_advantage, rewards])
            else:
                history = self._model_train.fit(
                    x = input,
                    y = [action_advantage, rewards],
                    batch_size = self.nr_snakes,
                    epochs = 1,
                    shuffle = True,
                    verbose = 0
                )
                history = history.history
                # print(history)
                loss = [history["loss"][-1],
                        history["output_1_loss"][-1],
                        history["output_2_loss"][-1]]
            if debug and DEBUG_TRAIN and dii == i:
                logits, _, values = self._model_train.action_value(input[dj:dj+1])
                p = np.exp(logits[dj])/sum(np.exp(logits[dj]))
                print("Value  After Train", values[dj])
                print("Logits After Train", logits[dj], "p", p)
            del input
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


    def move_select(self, move_result, display):
        debug = self.debug

        if False and self.frame() % self._switch_period == 0:
            self._model_run, self._model_train = self._model_train, self._model_run
        head  = self.head()
        if self._channels == 1 or DEBUG_INPUT:
            apple = self.apple()
            tail  = self.tail()

        h  = self.frame_history_now()
        h0 = self.frame_history_then()

        batch_size = 2048
        di, dj = divmod(self._debug_index, batch_size)
        dii = di * batch_size
        all_actions = []
        all_values  = []
        for i in range(0, self.nr_snakes, batch_size):
            i1 = min(i+batch_size, self.nr_snakes)
            if self._channels == 1 or DEBUG_INPUT:
                input = [[self._field0[j],
                          self._point_image[apple[j]],
                          self._point_image[head[j]],
                          self._point_image[tail[j]]
                ] for j in range(i, i1)]
                input = np.array(input, dtype=TYPE_FLOAT)
                # Tensorflow CPU cannot handle channel first (GPU can)
                # Move channels to the end (this doesn't copy)
                # (of course it will slow down the copy to tensor)
                input = np.rollaxis(input, 1,4)
                # input = tf.convert_to_tensor(input, dtype=tf.float32)
                if DEBUG_INPUT:
                    self.input_equal(i, input, self._deep_field0[i:i1])
            else:
                input = self._deep_field0[i:i1]
            if DEBUG_INPUT_PRINT and dii == i:
                print("NOW")
                print(input[dj,:,:,CHANNEL_BODY].astype(np.int8))
                print(input[dj,:,:,CHANNEL_HEAD].astype(np.int8))
                print(input[dj,:,:,CHANNEL_TAIL].astype(np.int8))
                print(input[dj,:,:,CHANNEL_APPLE].astype(np.int8))
            # The returned values are of type numpy.ndarray
            # Except logits which is of type tf.Tensor as long as
            # action_value uses predict_on_batch() instead of predict()
            logits, actions, values = self._model_run.action_value(input)
            del input
            # print(type(logits), type(actions), type(values))
            if debug and dii == i:
                p = np.exp(logits[dj])/sum(np.exp(logits[dj]))
                print("Logits", logits[dj], "p", p)
            del logits
            # print(actions, values)
            all_actions.append(actions)
            all_values.append(values)

        all_values  = np.concatenate(all_values,  axis=None)
        all_actions = np.concatenate(all_actions, axis=None)

        if h0 is not None:
            #reward_moves = np.random.uniform(
            #    self._rewards.move - self._rewards.rand /2,
            #    self._rewards.move + self._rewards.rand /2)
            #reward_moves = np.random.uniform(
            #    (reward_moves-self._rewards.rand/2) * self._history,
            #    (reward_moves+self._rewards.rand/2) * self._history,
            #    size=self.nr_snakes)
            reward_moves = self._rewards.move * self._history
            rewards = self._history_gained * self._rewards.apple
            rewards += np.where(self._history_game0 == self._nr_games,
                                # Bootstrap from discounted best estimate
                                (all_values + reward_moves) * self._discount,
                                self._rewards.crash)

            if NORMALIZE and self.nr_snakes > 1:
                mean = rewards.mean()
                var = rewards.var()
                var  = math.sqrt(var * self.nr_snakes / (self.nr_snakes-1))
                reward = (rewards - mean) / var
            # Calculate the advantage = current reward - predicted reward
            advantage = rewards - self._old_values[h0]

            old_action = self._old_action[h0]

            if debug:
                print("Old Action = %d, Old Value = %f, New Value = %f" %
                      (old_action[self._debug_index],
                       self._old_values[h0][self._debug_index],
                       all_values[self._debug_index]))
                print("Reward = %f, Advantage = %f (Old Game = %d, New Game = %d)" %
                      (rewards[self._debug_index],
                       advantage[self._debug_index],
                       self._history_game0[self._debug_index],
                       self._nr_games[self._debug_index]))
            self.fit(old_action, advantage, rewards)

        self._old_action[h] = all_actions
        self._old_values[h] = all_values
        if debug:
            print("New Value = %f, New Action = %d" % (all_values[self._debug_index], all_actions[self._debug_index]))
        pos = head + self.DIRECTIONS[all_actions]
        return pos
