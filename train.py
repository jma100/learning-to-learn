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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util
import time as tm
import datetime
import numpy as np

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 100, "Log period.")
flags.DEFINE_integer("evaluation_period", 1000, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")
flags.DEFINE_string("events_path", None, "Path for saved checkpoints.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")


def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  if FLAGS.save_path is not None:
    if os.path.exists(FLAGS.save_path):
      raise ValueError("Folder {} already exists".format(FLAGS.save_path))
    else:
      os.mkdir(FLAGS.save_path)

  if FLAGS.events_path is not None:
    if os.path.exists(FLAGS.events_path):
      raise ValueError("Folder {} already exists".format(FLAGS.events_path))
    else:
      os.mkdir(FLAGS.events_path)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)

  # Optimizer setup.
  optimizer = meta.MetaOptimizer(**net_config)
  minimize = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)
  step, update, reset, cost_op, _ = minimize

  # Create histograms of all trainable variables
       # tf.trainable_variables()
       # [<tf.Variable 'vars_optimizer/cw_deep_lstm/lstm_1/w_gates:0' shape=(22, 80) dtype=float32_ref>,
       # <tf.Variable 'vars_optimizer/cw_deep_lstm/lstm_1/b_gates:0' shape=(80,) dtype=float32_ref>,
       # <tf.Variable 'vars_optimizer/cw_deep_lstm/lstm_2/w_gates:0' shape=(40, 80) dtype=float32_ref>,
       # <tf.Variable 'vars_optimizer/cw_deep_lstm/lstm_2/b_gates:0' shape=(80,) dtype=float32_ref>,
       # <tf.Variable 'vars_optimizer/cw_deep_lstm/linear/w:0' shape=(20, 1) dtype=float32_ref>,
       # <tf.Variable 'vars_optimizer/cw_deep_lstm/linear/b:0' shape=(1,) dtype=float32_ref>]
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name[:-2], var)

  #merge all summaries into a single "operation" which we can execute in a session
  summary_op = tf.summary.merge_all()

  current_time = tm.strftime("%Y_%m_%d-%H:%M:%S")
  logs_path = os.path.join(FLAGS.events_path, current_time)

  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0

    writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
    global_start_time = tm.time()
    summary_file = open(os.path.join(FLAGS.events_path, "results.txt"), "w") 

    errors = [-5,-4,-3,-2,-1]
    converged = False
    eps = 0.01
    steps = 5
    step_converged = -1

    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost, summary = util.run_epoch(sess, cost_op, summary_op, [update, step], reset,
                                  num_unrolls)
      total_time += time
      total_cost += cost


      # write summary every log period
      # Logging.
      if (e + 1) % FLAGS.log_period == 0:
        writer.add_summary(summary, e)
        print("Training time so far(HOUR:MIN:SEC):   "+str(datetime.timedelta(seconds=int(tm.time()-global_start_time))))

        util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         FLAGS.log_period)
        total_time = 0
        total_cost = 0

      # Evaluation.
      if (e + 1) % FLAGS.evaluation_period == 0:
        eval_cost = 0
        eval_time = 0
        for _ in xrange(FLAGS.evaluation_epochs):
          time, cost, _ = util.run_epoch(sess, cost_op, summary_op, [update], reset,
                                      num_unrolls)
          eval_time += time
          eval_cost += cost

        # Record classification error
        error = np.log10(eval_cost / FLAGS.evaluation_epochs)
        error_summary = tf.Summary()
        error_summary.value.add(tag='log error (training time)', simple_value=error)
        writer.add_summary(error_summary, e)
        summary_file.write("Timestep:"+str(e) + " "+ "Error:"+str(error))

        # Find convergence time step
        if not converged:
          errors.append(error)
          _ = errors.pop(0)
          total = 0
          for i in range(-1,steps-1):
            total += abs(errors[i+1]-errors[i])
          if total/float(steps) <= eps:
#        if abs(errors[-1])<0.25:
            converged = True
            step_converged = e
            summary_file.write("Converged at step " + str(e))
            print('converged at step ' + str(e))
            print(errors)

        util.print_stats("EVALUATION", eval_cost, eval_time,
                         FLAGS.evaluation_epochs)

        if FLAGS.save_path is not None and eval_cost < best_evaluation:
          print("Removing previously saved meta-optimizer")
          for f in os.listdir(FLAGS.save_path):
            os.remove(os.path.join(FLAGS.save_path, f))
          print("Saving meta-optimizer to {}".format(FLAGS.save_path))
          optimizer.save(sess, FLAGS.save_path)
          best_evaluation = eval_cost
    if not converged:
      summary_file.write("Converged at step " + str(step_converged))

    writer.close()
    summary_file.close()

if __name__ == "__main__":
  tf.app.run()
