from PokerRL.game.InteractiveGame import InteractiveGame
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

if __name__ == '__main__':
    eval_agent = EvalAgentDeepCFR.load_from_disk(
        path_to_eval_agent="/home/asellerg/data/poker_ai_data/eval_agent/DeepCFR_NLHE_6-max_trial_2/15/eval_agentAVRG_NET.pkl")

    game = InteractiveGame(env_cls=eval_agent.env_bldr.env_cls,
                           env_args=eval_agent.env_bldr.env_args,
                           seats_human_plays_list=[0],
                           eval_agent=eval_agent,
                           )

    game.start_to_play()