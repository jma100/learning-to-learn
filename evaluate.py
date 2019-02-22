# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util
import time as tm
import datetime
import numpy as np
import os

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 100, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")
flags.DEFINE_string("eval_path", None, "Path for saved evaluation checkpoints.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")


def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps

  if FLAGS.eval_path is not None:
    if os.path.exists(FLAGS.eval_path):
      raise ValueError("Folder {} already exists".format(FLAGS.eval_path))
    else:
      os.mkdir(FLAGS.eval_path)

  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem,
                                                         FLAGS.path)

  # Optimizer setup.
  if FLAGS.optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "L2L":
    if FLAGS.path is None:
      logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(**net_config)
    meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
    _, update, reset, cost_op, _ = meta_loss
  else:
    raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

  current_time = tm.strftime("%Y_%m_%d-%H:%M:%S")
  logs_path = os.path.join(FLAGS.eval_path, current_time)
  summary_op = tf.summary.merge_all()

  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
    global_start_time = tm.time()
    summary_file = open(os.path.join(FLAGS.eval_path, "results.txt"), "w") 

    total_time = 0
    total_cost = 0
    errors = [-1,1,-1,1,-1]
    converged = False
    eps = 0.0001
    steps = 5
    step_converged = -1
    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost, summary = util.run_epoch(sess, cost_op, summary_op, [update], reset,
                                  num_unrolls)
      writer.add_summary(summary, e)
      total_time += time
      total_cost += cost

      # Record classification error
      error = np.log10(total_cost / (e+1))
      error_summary = tf.Summary()
      error_summary.value.add(tag='classification error', simple_value=error)
      writer.add_summary(error_summary, e)
      summary_file.write("Timestep:"+str(e) + " "+ "Error:"+str(error))

      # Find convergence time step
      if not converged:
        errors.append(error)
        _ = errors.pop(0)
        total = 0
        for i in range(-1,steps-1):
          total += abs(errors[i+1]-errors[i])
#        if total/float(steps) <= eps:
        if abs(errors[-1])<0.25:
          converged = True
          step_converted = e
          summary_file.write("Converged at step " + str(e))
          print('converged at step ' + str(e))
          print(errors)

    # Results.
    util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                     total_time, FLAGS.num_epochs)
    print("Training time so far(HOUR:MIN:SEC):   "+str(datetime.timedelta(seconds=int(tm.time()-global_start_time))))

    if not converged:
      summary_file.write("Converged at step " + str(step_converged))

    writer.close()
    summary_file.close()

if __name__ == "__main__":
  tf.app.run()
