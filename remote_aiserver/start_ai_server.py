from algorithms.model.model import Model as Model
from config.config import Config
from aiserver import AIServer
import sys
# ports=Config.AISERVERPORT
if __name__ == '__main__':
    port=int(sys.argv[1])
    model_path= sys.argv[2]
    # "/reinforcement_platform/actor_platform/good_model"
    ai_server1 = AIServer(port, Model, model_path)
    ai_server1.prepare_connection()
    ai_server1.handle_request()