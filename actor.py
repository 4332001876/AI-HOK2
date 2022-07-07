# -*- coding: utf-8 -*-
"""
    KingHonour Data production process
"""
import os
import sys
import importlib
import traceback
from collections import deque
import numpy as np
from config.config import Config
from framework.common.common_func import CommonFunc
from framework.common.common_log import CommonLogger
from framework.common.common_log import g_log_time

from framework.common.common_func import log_time_func


import time

IS_TRAIN = Config.IS_TRAIN
LOG = CommonLogger.get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
OS_ENV = os.environ
IS_DEV = OS_ENV.get("IS_DEV")


class Actor:
    """
    used for sample logic
        run 1 episode
        save sample in sample manager
    """

    ALL_CONFIG_DICT = [
        [{"hero": "luna", "skill": "weak"} for _ in range(2)],
    ]
    HERO_DICT = {
        "luban": 112,
        "miyue": 121,
        "lvbu": 123,
        "libai": 131,
        "makeboluo": 132,
        "direnjie": 133,
        "guanyu": 140,
        "diaochan": 141,
        "luna": 146,
        "hanxin": 150,
        "huamulan": 154,
        "buzhihuowu": 157,
        "jvyoujing": 163,
        "houyi": 169,
        "zhongkui": 175,
        "ganjiangmoye": 182,
        "kai": 193,
        "gongsunli": 199,
        "peiqinhu": 502,
        "shangguanwaner": 513,
        }
    # def __init__(self, id, type):
    def __init__(self, id, agents, max_episode: int = 0,
                 env=None):
        self.m_config_id = id
        self.m_task_uuid = Config.TASK_UUID
        self.m_steps = Config.STEPS
        self.m_init_path = Config.INIT_PATH
        self.m_update_path = Config.UPDATE_PATH
        self.m_mem_pool_path = Config.MEM_POOL_PATH
        self.m_task_name = Config.TASK_NAME
        self.m_algorithm = Config.ALGORITHM

        # self.m_replay_buffer = deque()
        self.m_episode_info = deque(maxlen=100)
        # self.m_ip = CommonFunc.get_local_ip()
        self.env = env
        # TODO: set max episode
        self._max_episode = max_episode

        self.m_run_step = 0
        self.m_best_reward = 0

        self._last_print_time = time.time()
        self._episode_num = 0
        CommonLogger.set_config(self.m_config_id)
        # self.m_influxdb = InfluxTool(self.m_config_id)

        self.agents = agents

    def set_env(self, environment):
        self.env = environment

    def set_sample_managers(self, sample_manager):
        self.m_sample_manager = sample_manager

    def set_agents(self, agents):
        self.agents = agents

    def _get_common_ai(self, eval, load_models):
        use_common_ai = [False] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if eval:
                if load_models is None or len(load_models) < 2:
                    if not agent.keep_latest:
                        use_common_ai[i] = True
                elif load_models[i] is None:
                    use_common_ai[i] = True

        return use_common_ai

    def _run_episode(self, env_config, eval=False, load_models=None, eval_info=""):
        for item in g_log_time.items():
            g_log_time[item[0]] = []
        # alias
        sample_manager = self.m_sample_manager

        # LOG.info("latest_version : %d, cmap_list_size : %d, main_camp : %d"%(aiprocess._model_manager.latest_version,\
        #         len(self.m_env.camp_list), aiprocess._model_manager.main_camp))
        done = False

        log_time_func('reset')
        log_time_func('one_episode')
        LOG.debug("reset env")
        self.agents.reverse()
        use_common_ai = self._get_common_ai(eval, load_models)

        # ATTENTION: agent.reset() loads models from local file which cost a lot of time.
        #            Before upload your code, please check your code to avoid ANY time-wasting
        #            operations between env.reset() and env.close_game(). Any TIMEOUT in a round
        #            of game will cause undefined errors.

        # reload agent models
        for i, agent in enumerate(self.agents):
            LOG.debug("reset agent {}".format(i))
            if eval:
                if load_models is None or len(load_models) < 2:
                    agent.reset("common_ai")
                else:
                    if load_models[i] is None:
                        agent.reset("common_ai")
                    else:
                        agent.reset("network", model_path=load_models[i])
            else:
                if len(load_models) == 1 and not agent.keep_latest:
                    agent.reset(Config.ENEMY_TYPE, model_path=load_models[0])
                else:
                    agent.reset(Config.ENEMY_TYPE)

        render = self.render if eval else None
        # restart a new game
        # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
        o,r,d,state_dict = self.env.reset(env_config, use_common_ai=use_common_ai, eval=eval, render=render)
        if state_dict[0] is None:
            game_id = state_dict[1]['game_id']
        else:
            game_id = state_dict[0]['game_id']

        # update agents' game information
        for i, agent in enumerate(self.agents):
            player_id = self.env.player_list[i]
            camp = self.env.player_camp.get(player_id)
            agent.set_game_info(camp, player_id)

        # reset mem pool and models
        LOG.debug("reset sample_manager")
        sample_manager.reset(agents=self.agents, game_id=game_id)
        rewards = [[], []]
        step = 0
        log_time_func('reset', end=True)
        game_info = {}
        episode_infos = [{"h_act_num": 0} for _ in self.agents]

        while not done:
            log_time_func('one_frame')
            # while True:
            actions = []
            log_time_func('agent_process')
            for i, agent in enumerate(self.agents):
                if use_common_ai[i]:
                    actions.append(None)
                    rewards[i].append(0.)
                    continue

                action, d_action, sample = agent.process(state_dict[i])
                if eval:
                    action = d_action
                # print("input act: [{}], {}, {}".format(i, action, state_dict[i]["legal_action"][:12]))
                actions.append(action)
                if action[0] == 10:
                    episode_infos[i]['h_act_num'] += 1
                rewards[i].append(sample['reward'])

                if agent.is_latest_model and not eval:
                    sample_manager.save_sample(**sample, agent_id=i,
                            game_id=game_id, uuid=self.m_task_uuid)
            log_time_func('agent_process', end=True)

            log_time_func('step')
            # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
            o,r,d,state_dict = self.env.step(actions)
            # if np.isnan(r[0][-1]) or np.isnan(r[1][-1]):
            #     exit(0)
            log_time_func('step', end=True)

            req_pbs = self.env.cur_req_pb
            if req_pbs[0] is None:
                req_pb = req_pbs[1]
            else:
                req_pb = req_pbs[0]
            LOG.debug("step: {}, frame_no: {}, reward: {}, {}".format(step, req_pb.frame_state.frameNo,
                                                                           r[0], r[1]))
            step += 1
            done = d[0] or d[1]
            if req_pb.gameover:
                print("really gameover!!!")

            if done:
                for i, agent in enumerate(self.agents):
                    if agent.is_latest_model and not eval:
                        if state_dict[i]['reward']!=None:
                            if type(state_dict[i]['reward'])==tuple:
                                #if reward is a vec
                                sample_manager.save_last_sample(agent_id=i, reward=state_dict[i]['reward'][-1])
                            else:
                                #if reward is a float number
                                sample_manager.save_last_sample(agent_id=i, reward=state_dict[i]['reward'])
                        else:
                            sample_manager.save_last_sample(agent_id=i, reward=0)
            log_time_func('one_frame', end=True)

        self.env.close_game()

        game_info['length'] = req_pb.frame_no
        loss_camp = -1
        camp_hp = {}
        all_camp_list = []
        for npc_state in req_pb.frame_state.npc_states:
            if npc_state.actor_type == 2 and npc_state.sub_type == 24:
                if npc_state.hp <= 0:
                    loss_camp = npc_state.camp
                camp_hp[npc_state.camp] = npc_state.hp
                all_camp_list.append(npc_state.camp)
            if npc_state.actor_type == 2 and npc_state.sub_type in [21, 24]:
                LOG.info("Tower {} in camp {}, hp: {}".format(npc_state.sub_type, npc_state.camp, npc_state.hp))

        for i, agent in enumerate(self.agents):
            if use_common_ai[i]:
                continue
            for hero_state in req_pbs[i].frame_state.hero_states:
                if agent.player_id == hero_state.actor_state.runtime_id:
                    episode_infos[i]['money_pre_frame'] = hero_state.moneyCnt / game_info['length']
                    episode_infos[i]['kill'] = hero_state.killCnt
                    episode_infos[i]['death'] = hero_state.deadCnt
                    episode_infos[i]['hurt_pre_frame'] = hero_state.totalHurt / game_info['length']
                    episode_infos[i]['hurtH_pre_frame'] = hero_state.totalHurtToHero / game_info['length']
                    episode_infos[i]['hurtBH_pre_frame'] = hero_state.totalBeHurtByHero / game_info['length']
                    episode_infos[i]['hero_id']=self.HERO_DICT[env_config[0]["hero"]]
                    break
            if loss_camp == -1:
                episode_infos[i]['win'] = 0
            else:
                episode_infos[i]['win'] = -1 if agent.hero_camp == loss_camp else 1

            # print("rewards {} :".format(i),rewards[i])
            episode_infos[i]['reward'] = np.sum(rewards[i])
            episode_infos[i]['h_act_rate'] = episode_infos[i]['h_act_num'] / step

        if IS_TRAIN and not eval:
            LOG.debug("send sample_manager")
            sample_manager.send_samples()
            LOG.debug("send done.")

        log_time_func('one_episode', end=True)
        # print game information
        self._print_info(game_id, game_info, episode_infos, eval, eval_info, common_ai=use_common_ai)

    def _print_info(self, game_id, game_info, episode_infos, eval, eval_info="", common_ai=None):
        if common_ai is None:
            common_ai = [False] * len(self.agents)
        # TODO: update this code, and add some details about reward.
        if eval and len(eval_info) > 0:
            LOG.info("eval_info: %s" % eval_info)
        LOG.info('=' * 50)
        LOG.info("game_id : %s"%game_id)
        for item in g_log_time.items():
            if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
                continue
            mean = np.mean(item[1])
            max = np.max(item[1])
            sum = np.sum(item[1])
            LOG.info('%s | sum: %s mean:%s max:%s times:%s' % (item[0], sum, mean, max, len(item[1])))
            g_log_time[item[0]] = []
        LOG.info('=' * 50)
        # TODO: try to get common_ai information!
        for i, agent in enumerate(self.agents):
            if common_ai[i]:
                continue
            LOG.info("Agent is_main:{}, type:{}, camp:{},reward:{:.3f}, win:{}, win_{}:{},h_act_rate:{}"
                     .format(agent.keep_latest and eval, agent.agent_type, agent.hero_camp,
                            episode_infos[i]['reward'], episode_infos[i]['win'],episode_infos[i]['hero_id'],episode_infos[i]['win'],
                            1.0))
            LOG.info("Agent is_main:{}, money_pre_frame:{:.2f}, kill:{}, death:{}, hurt_pf:{:.2f}"
                     .format(agent.keep_latest and eval,
                               episode_infos[i]['money_pre_frame'],
                               episode_infos[i]['kill'],
                               episode_infos[i]['death'],
                               episode_infos[i]['hurt_pre_frame']))
        LOG.info("game info length:{}".format(game_info['length']))

        LOG.info('=' * 50)

    def run(self, eval_mode=False, eval_number=-1, load_models=[]):
        from hok import GameRender
        if eval_mode or self.m_config_id == 0:
            self.render = GameRender()
        else:
            self.render = None

        self._last_print_time = time.time()
        self._episode_num = 0
        MAX_REPEAT_ERR_NUM = 2
        repeat_num = MAX_REPEAT_ERR_NUM
        # SKILL_DICT = {"heal": 80102, "rage": 80110, "flash": 80115, "sprint": 80109,
        #               "execute": 80108, "disrupt": 80105, "daze": 80103, "purity": 80107,
        #               "weak": 80121}



        # config_dicts = [{"hero": "diaochan", "skill": "rage"} for _ in range(2)]
        if eval_mode:
            LOG.info("eval_mode start...")
            agent_0, agent_1 = 0, 1
            cur_models = [load_models[agent_0], load_models[agent_1]]
            cur_eval_cnt = 1
            swap = False

        last_clean = time.time()

        while True:
            hero_index=os.getenv("hero_index")
            if hero_index=="all":
                config_dicts = self.ALL_CONFIG_DICT[np.random.randint(0,len(self.ALL_CONFIG_DICT))]
            else:
                config_dicts = self.ALL_CONFIG_DICT[0]

                # config_dicts = self.ALL_CONFIG_DICT[int(hero_index)]
            try:
                # provide a init eval value at the first episode
                if eval_mode:
                    if swap:
                        eval_info = "{} vs {}, {}/{}".format(agent_1, agent_0, cur_eval_cnt, eval_number)
                    else:
                        eval_info = "{} vs {}, {}/{}".format(agent_0, agent_1, cur_eval_cnt, eval_number)

                    self._run_episode(config_dicts, True, load_models=cur_models,
                                      eval_info=eval_info)
                    # swap camp
                    cur_models.reverse()
                    swap = not swap
                else:
                    eval_with_common_ai = (self._episode_num+0) % Config.EVAL_FREQ == 0 and self.m_config_id == 0
                    self._run_episode(config_dicts, eval_with_common_ai, load_models=load_models)
                if self.env.render is not None:
                    self.env.render.dump_one_round()
                self._episode_num += 1
                repeat_num = MAX_REPEAT_ERR_NUM
            except Exception as e:
                LOG.error("_run_episode err: {}/{}".format(repeat_num, MAX_REPEAT_ERR_NUM))
                LOG.error(e)
                traceback.print_exc()
                # traceback.print_tb(e.__traceback__)
                repeat_num -= 1
                if repeat_num == 0:
                    raise e

            if eval_mode:
                # update eval agents and eval cnt
                cur_eval_cnt += 1
                if cur_eval_cnt > eval_number:
                    cur_eval_cnt = 1

                    agent_1 += 1
                    if agent_1 >= len(load_models):
                        agent_0 += 1
                        agent_1 = agent_0+1

                    if agent_1 >= len(load_models):
                        # eval end
                        break

                    cur_models = [load_models[agent_0], load_models[agent_1]]
            else:
                # In training process, clean game_log every 1 hour.
                # feel free to DIY it by yourself.
                now = time.time()
                CLEAN_INTERVAL=60*60
                if self.m_config_id == 0 and now - last_clean > CLEAN_INTERVAL:
                    LOG.info("Clean the game_log automatically.")
                    os.system("find /reinforcement_platform/actor_platform/game_log -mmin +60 -name \"*\" -exec rm -rfv {} \;")
                    last_clean = now

            if 0 < self._max_episode <= self._episode_num:
                break

        for agent in self.agents:
            agent.close()