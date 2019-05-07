import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        #Initial
        self.init_pos = init_pose if init_pose is not None else np.array([0., 0., 5.])
        print(self.init_pos)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        dist_diff_z = 1.-.004*(abs(self.sim.pose[2] - self.target_pos[2]))
        
        dist_diff_x = 1.-.004*(abs(self.sim.pose[0] - self.target_pos[0]))
        
        dist_diff_y = 1.-.004*(abs(self.sim.pose[1] - self.target_pos[1]))

        """
        reward_dist_diff = 1.-.004*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        reward_vel_dist = 1. - .003*(np.abs(5*self.sim.v[2] - (np.abs(self.sim.v[0] + self.sim.v[1]))))
       
        reward = reward_dist_diff + reward_vel_dist
        
        """
        reward = 2*dist_diff_z + dist_diff_x + dist_diff_y
        
        """
        print("r_dist_diff: {:4.2f}".format(reward))
        """
        if(self.sim.pose[2] >= (.2*self.target_pos[2])):
            reward += (dist_diff_z*100.)
        if(self.sim.pose[2] < self.init_pos[2]):
            reward -= 15.
            
        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state