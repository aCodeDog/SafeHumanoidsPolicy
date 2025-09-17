import torch
from envs.g1_with_hand_robot import G1HandRobot
from envs.g1_with_inspire_config import G1InspireCfg 
from utils.utils import class_to_dict

class G1HandEnv(G1HandRobot):
    def __init__(self, cfg: G1InspireCfg, sim_params, physics_engine, sim_device, headless):
        """ Simplified constructor - all functionality moved to parent class """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

def main():
    """Main function to run the sim2sim environment"""
    env = G1HandEnv(G1InspireCfg, class_to_dict(G1InspireCfg.sim), "genesis", 'cuda', False)

    # Load policy
    theta_attach_yaw = "assets/policy.jit"
    policy = torch.jit.load(theta_attach_yaw)
    policy.to(device="cuda")
    
    # Reset environment and run simulation
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()