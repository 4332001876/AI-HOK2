sleep 2s
export GC_MODE="local"
cd /root
sh deploy_gamecore.sh
cd /reinforcement_platform/actor_platform/code
python gen_transfer_evalwith_common_script.py ./models
rm ../log/info*.log
ps -aux |grep sgame_simulator_ |grep -v grep |awk '{print $2}' |xargs kill -9
ps -aux |grep entry.py |grep -v grep |awk '{print $2}' |xargs kill -9
sleep 2s
for(( i=0;i<8;i++ ))
do
nohup sh eval-${i}.sh >>actor_$i.log &
done
