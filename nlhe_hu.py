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
    exp = "SD-CFR_NLHE_HU_trial_10"
    ctrl = Driver(t_prof=TrainingProfile(name=exp,
                                         nn_type="recurrent",
                                         eval_agent_export_freq=5,  # export API to play against the agent
                                         checkpoint_freq=5,

                                         n_actions_traverser_samples=None,  # = external sampling in FHP
                                         n_traversals_per_iter=5000,
                                         n_batches_adv_training=1000,
                                         mini_batch_size_adv=512,  # *8
                                         init_adv_model="random",

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         max_buffer_size_adv=1e6,  # *8 LAs
                                         lr_adv=0.002,
                                         lr_patience_adv=99999999,  # No lr decay

                                         n_batches_avrg_training=1000,
                                         mini_batch_size_avrg=1024,  # *8
                                         init_avrg_model="random",

                                         use_pre_layers_avrg=True,
                                         n_cards_state_units_avrg=192,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_avrg=64,

                                         max_buffer_size_avrg=1e6,
                                         lr_avrg=0.002,
                                         lr_patience_avrg=99999999,  # No lr decay

                                         game_cls=DiscretizedNLHoldem,
                                         n_seats=2,
                                         agent_bet_set=bet_sets.B_3,

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET, # DeepCFR
                                         ),
                                         path_data="/home/asellerg/data/poker_ai_data/",
                                         DISTRIBUTED=True,
                                         CLUSTER=False,
                                         n_learner_actor_workers=8,
                                         lbr_args=LBRArgs(
                                            n_lbr_hands_per_seat=50000,
                                            DISTRIBUTED=True,
                                            lbr_bet_set=bet_sets.B_3,
                                            lbr_check_to_round=Poker.TURN,
                                            n_parallel_lbr_workers=10),
                                         ),
                  eval_methods={
                    "lbr": 4
                  },
                  n_iterations=15,
                  name_to_import="SD-CFR_NLHE_HU_trial_10",
                  iteration_to_import=25)
    ctrl.run()
