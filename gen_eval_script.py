import  sys
import os
if __name__ == '__main__':
    dst_path=sys.argv[1]
    file_list=os.listdir(dst_path)
    file_list=sorted(file_list)
    current_path=os.path.abspath(dst_path)
    for i in range(0,8):
        script_path = "./eval-{}.sh".format(i)
        script_file = open(script_path, "w")
        for file_name in file_list:
            model_path=os.path.join(current_path,file_name)
            if file_name.endswith("tar"):
                os.system("cd {} ;tar xvf {}".format(current_path,model_path))
                if os.path.exists(model_path):
                    os.system("rm -r  {}".format(model_path))
                script_file.write("python entry.py --actor_id={} --eval_number=2 --agent_models=\"common_ai,{}\"\n".format(i,os.path.join(current_path, file_name[:-4])))
            else:
                script_file.write("python entry.py --actor_id={} --eval_number=2 --agent_models=\"common_ai,{}\"\n".format(i,os.path.join(current_path, file_name)))

