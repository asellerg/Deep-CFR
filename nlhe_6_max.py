import ray
import time

from PokerRL.game.Poker import Poker
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLHoldem

from PokerRL.eval.lbr.LBRArgs import LBRArgs

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver



if __name__ == '__main__':
    # Need to comment out PokerRL imports in poker_ai and return ACTIONS for legal_actions.
    exp = "DeepCFR_NLHE_6-max_trial_73_no_all_in_max_3_raises"
    ctrl = Driver(t_prof=TrainingProfile(name=exp,
                                         nn_type="recurrent",
                                         eval_agent_export_freq=1,  # export API to play against the agent
                                         checkpoint_freq=2,
                                         start_chips=10000,
                                         chip_randomness=(-5000, 10000),

                                         n_actions_traverser_samples=None,
                                         n_traversals_per_iter=50000,
                                         n_batches_adv_training=1000,
                                         mini_batch_size_adv=512,  # *8
                                         init_adv_model="random",

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=512,
                                         n_merge_and_table_layer_units_adv=128,
                                         n_units_final_adv=128,
                                         rnn_units_adv=256,
                                         rnn_stack_adv=3,

                                         max_buffer_size_adv=2e5,  # *8 LAs
                                         lr_adv=0.005,
                                         lr_patience_adv=99999999,  # No lr decay

                                         n_batches_avrg_training=2000,
                                         mini_batch_size_avrg=1024,  # *8
                                         init_avrg_model="random",

                                         use_pre_layers_avrg=True,
                                         n_cards_state_units_avrg=512,
                                         n_merge_and_table_layer_units_avrg=128,
                                         n_units_final_avrg=128,
                                         rnn_units_avrg=128,
                                         rnn_stack_avrg=3,

                                         max_buffer_size_avrg=2e5,
                                         lr_avrg=0.002,
                                         lr_patience_avrg=99999999,  # No lr decay

                                         game_cls=DiscretizedNLHoldem,
                                         n_seats=6,
                                         agent_bet_set=bet_sets.B_3_NO_ALL_IN,

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET, # DeepCFR
                                         ),
                                         path_data="/home/asellerg/data/poker_ai_data/",
                                         DISTRIBUTED=True,
                                         CLUSTER=False,
                                         n_learner_actor_workers=8),
                  eval_methods={
                  },
                  n_iterations=4)
                  # iteration_to_import=12,
                  # name_to_import="DeepCFR_NLHE_6-max_trial_71_no_all_in")
    ctrl.run()
