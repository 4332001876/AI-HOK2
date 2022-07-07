import  sys
import os
import time
if __name__ == '__main__':
    dst_path=sys.argv[1]
    file_list=os.listdir(dst_path)
    file_list=sorted(file_list)
    current_path=os.path.abspath(dst_path)
    battle_num=2
    for i in range(len(file_list)):
        for j in range(len(file_list)):
            if i!=j:
                model0_path=os.path.join(current_path,file_list[i])
                model1_path=os.path.join(current_path,file_list[j])
                print("start battle {} {}".format(model0_path,model1_path))
                _cmd="cp {} /reinforcement_platform/actor_platform/code/algorithms/model/model0/".format(model0_path)
                print(_cmd)
                os.system(_cmd)
                _cmd="cd /reinforcement_platform/actor_platform/code/algorithms/model/model0/;tar xvf {};mv checkpoints*/* ckpt/; rm -r checkpoints*".format(file_list[i])
                print(_cmd)
                os.system(_cmd)
                _cmd="cp {} /reinforcement_platform/actor_platform/code/algorithms/model/model1/".format(model1_path)
                print(_cmd)
                os.system(_cmd)
                _cmd="cd /reinforcement_platform/actor_platform/code/algorithms/model/model1/;tar xvf {};mv checkpoints*/* ckpt/;rm -f checkpoints*".format(file_list[j])
                print(_cmd)
                os.system(_cmd)
                _cmd="cd /reinforcement_platform/actor_platform/code;nohup sh start_battle.sh 0 2 \"gameid-2021-{}-{}\" \" \" \" \" 2>&1 &".format(i,j)
                print(_cmd)
                os.system(_cmd)
                while True :
                    _cmd="grep -nr \"win\" /reinforcement_platform/actor_platform/code/game_log/gameid-2021-{}-{}-battle.log |wc -l".format(i,j)
                    print(_cmd)
                    r=os.popen(_cmd)
                    n=int(str(r.read().strip()))
                    print(n)
                    time.sleep(3)
                    if int(str(n))>=battle_num:
                        break
