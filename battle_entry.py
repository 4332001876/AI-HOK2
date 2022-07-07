import sys
sys.path.append('.')
# sys.path.append('./gamecore/kinghonour/proto_king/')
# sys.path.append('./lib')
from remote_aiserver.aiserver import AIServer
from absl import app as absl_app
from absl import flags
import random
import threading
import multiprocessing
import sail.common.logging as LOG
import time
import json
from battle_actor import BattleActor,RemoteAiServer
#from agent import RandomAgent as Agent
from agent import Agent as Agent
#from sample.sample_manager import DummySampleManager as SampleManager
from algorithms.model.sample_manager import SampleManager as SampleManager
# from predict.model import Model
from algorithms.model.model import Model
# from algorithms.model.model import Model1 as Model1
# from algorithms.model2.model import Model2 as Model2
from config.config import Config
from remote_aiserver.aiserver import AIServer
# import check
battle_json_file="./battle.json"
battle_dict=[]
# if not check.check_aiserver(battle_dict):
#     exit(0)
# json_str=json.dumps(battle_dict)
# with open(battle_json_file,"w") as f:
#     f.write(json_str)
# f.close()
FLAGS = flags.FLAGS
flags.DEFINE_string("gamecore_path", "~/.hok", "installation path of gamecore")
flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_string("gamecore_ip", "localhost", "address of gamecore")
flags.DEFINE_integer("thread_num", 1, "thread_num")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")
flags.DEFINE_integer("battle_number", 1, "battle number for evaluation")
flags.DEFINE_boolean("single_test", 0, "test_mode")
flags.DEFINE_boolean("battle_test", 0, "test_mode")
flags.DEFINE_string("hero", "", "hero")
flags.DEFINE_string("game_id", "", "game_id")
flags.DEFINE_string("game_log_path", "./game_log", "log path for game information")

MAP_SIZE = 100
AGENT_NUM = 2

# gamecore as lib
def gc_as_lib(argv):
    # TODO: used for different process
    # from gamecore.kinghonour.gamecore_client import GameCoreClient as Environment
    from hok import HoK1v1 as HoK1v1
    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.actor_id + thread_id
    agents = []
    game_id_init = "None"
    load_models = []
    for m in FLAGS.agent_models.split(','):
        if len(m) > 0:
            load_models.append(m)
    if FLAGS.single_test:
        Config.SINGLE_TEST = True
    print(load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None

    env = HoK1v1.load_game(runtime_id=actor_id,
                           gamecore_path=FLAGS.gamecore_path, game_log_path=FLAGS.game_log_path,
                           eval_mode=True, config_path="config.dat", remote_param=None)

    agents.append(RemoteAiServer("127.0.0.1",Config.AISERVERPORT[0]))
    agents.append(RemoteAiServer("127.0.0.1",Config.AISERVERPORT[1]))

    assert FLAGS.hero is not None
    sample_manager = SampleManager(mem_pool_addr=FLAGS.mem_pool_addr, mem_pool_type="mcp++",
                                   num_agents=AGENT_NUM, game_id=game_id_init, local_mode=True)
    actor = BattleActor(id=actor_id, agents=agents, hero=FLAGS.hero)
    actor.set_sample_managers(sample_manager)
    actor.set_env(env)
    actor.run(mode=Config.BATTLE_MODE, eval_number=FLAGS.battle_number,game_id=FLAGS.game_id,battle_dict=battle_dict)
    json_str=json.dumps(battle_dict)
    with open("battle.json","w") as f:
        f.write(json_str)
    f.close()
if __name__ == '__main__':
    absl_app.run(gc_as_lib)






