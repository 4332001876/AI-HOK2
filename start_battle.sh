hero=$1
eval_number=$2
game_id=$3
url1=$4
url2=$5
path=`pwd`
export GC_MODE=local
echo "gamecore initing"
cd ~ && sh deploy_gamecore.sh  
cd $path
echo "start download model1: $url1"
rm code1.tgz
wget $url1 -O code1.tgz && sleep 1s
tar xvf code1.tgz
rm -rf ./remote_aiserver/aiserver1
mv code ./remote_aiserver/aiserver1
#
echo "start download model2: $url2"
rm code2.tgz
wget $url2 -O code2.tgz && sleep 1s
tar xvf code2.tgz
rm -rf ./remote_aiserver/aiserver2
mv code ./remote_aiserver/aiserver2
ps -aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
ps -aux | grep sgame_sim | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 3s
cp remote_aiserver/aiserver.py remote_aiserver/aiserver1/aiserver.py
cp remote_aiserver/aiserver.py remote_aiserver/aiserver2/aiserver.py
cp remote_aiserver/start_ai_server.py remote_aiserver/aiserver1/start_ai_server.py
cp remote_aiserver/start_ai_server.py remote_aiserver/aiserver2/start_ai_server.py
echo "start ai sever"
model_dir="/reinforcement_platform/actor_platform/code/remote_aiserver"
cd remote_aiserver/aiserver1/ && nohup python3 start_ai_server.py 10010 "$model_dir/aiserver1/algorithms/checkpoint" > aiserver1.log 2>&1 &
cd remote_aiserver/aiserver2/ && nohup python3 start_ai_server.py 10011 "$model_dir/aiserver2/algorithms/checkpoint" > aiserver2.log 2>&1 &

sleep 10s
echo "start battle"
python battle_entry.py --battle_test  \
                --gamecore_ip="127.0.0.1" \
                --hero=$hero \
                --game_id=$game_id \
                --battle_number=$eval_number

sleep 10s
ps -aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
