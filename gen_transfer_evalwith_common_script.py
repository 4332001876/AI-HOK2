import  sys
import os
if __name__ == '__main__':

    dst_path=sys.argv[1]
    dst_path = os.path.abspath(dst_path)
    diaochan_model_dir=os.path.join(dst_path,"diaochan","model")
    luban_model_dir=os.path.join(dst_path,"luban","model")
    lvbu_model_dir=os.path.join(dst_path,"lvbu","model")
    miyue_model_dir=os.path.join(dst_path,"miyue","model")
    libai_model_dir=os.path.join(dst_path,"libai","model")

    print(diaochan_model_dir)

    # file_list=os.listdir(dst_path)
    # file_list=sorted(file_list)
    # current_path=os.path.abspath(dst_path)
    hero_index=str(os.getenv("hero_index"))
    for i in range(0,8):
        script_path = "./eval-{}.sh".format(i)
        script_file = open(script_path, "w")
        script_file.write("python entry.py --actor_id={} --eval_number=20 --agent_models=\"{},{}\"\n".format(i,diaochan_model_dir,"common_ai"))
        # if hero_index=="0":
        #
        #     script_file.write("python entry.py --actor_id={} --eval_number=20 --agent_models=\"{},{}\"\n".format(i,diaochan_model_dir,"common_ai"))
        # if hero_index=="1":
        #     script_file.write("python entry.py --actor_id={} --eval_number=20 --agent_models=\"{},{}\"\n".format(i,diaochan_model_dir,"common_ai"))
        # if hero_index=="2":
        #     script_file.write("python entry.py --actor_id={} --eval_number=20 --agent_models=\"{},{}\"\n".format(i,diaochan_model_dir,"common_ai"))
        # if hero_index=="3":
        #     script_file.write("python entry.py --actor_id={} --eval_number=20 --agent_models=\"{},{}\"\n".format(i,diaochan_model_dir,"common_ai"))
        # if hero_index=="4":
        #     script_file.write("python entry.py --actor_id={} --eval_number=20 --agent_models=\"{},{}\"\n".format(i,diaochan_model_dir,"common_ai"))

        script_file.close()