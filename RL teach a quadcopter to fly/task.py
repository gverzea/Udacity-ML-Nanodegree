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

        # Goal - quadcopter should lower from start coordinates 0,0,10 to 0,0,8, and then hover in position.
        # the episode ends when the time expires or when the quadcopter goes out-of-bounds.
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 8.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # big reward if the target location (with some error margin) has been reached
        # bad reward if the drone gets very close to crashing
        #print("actual position vs target", self.sim.pose[:3], self.target_pos)
        if np.all(abs(self.sim.pose[:3] - self.target_pos) <= 0.3) :
            reward = 100
        elif self.sim.pose[2] < 1 :
            reward = -100
        else:
            reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        #stopping condition - if target position has been reached
        if np.all(abs(self.sim.pose[:3] - self.target_pos) <= 0.3):
            print("target reached, at position ", self.sim.pose[:3])
            #done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state