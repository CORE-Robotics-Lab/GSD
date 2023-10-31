
import os
import numpy as np
from gym import utils as gymutils
from gym.envs.robotics import fetch_env
from gym.envs.robotics import rotations, utils


class FetchPushSingle(fetch_env.FetchEnv, gymutils.EzPickle):
    def __init__(self, reward_type=''):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_env.FetchEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__), "assets/robotics/fetch/pushsingle.xml"),
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.22,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.0,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        gymutils.EzPickle.__init__(self, reward_type=reward_type)
        self.init_obs = None
        assert self.has_object

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        ref_qpos = self.sim.data.get_joint_qpos("object0:joint")
        ref_qpos[:2] = self.initial_gripper_xpos[:2]
        offset = -0.15
        obj0_qpos = ref_qpos.copy()
        obj0_qpos[:2] += np.array([0, 1])*offset
        self.sim.data.set_joint_qpos("object0:joint", obj0_qpos)

        self.sim.forward()
        self.init_obs = self._get_obs()
        return True

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        obj0_pos = self.sim.data.get_site_xpos("object0")
        obj0_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        obj0_velp = self.sim.data.get_site_xvelp("object0") * dt
        obj0_velr = self.sim.data.get_site_xvelr("object0") * dt

        obj0_rel_pos = obj0_pos - grip_pos
        obj0_rel_velp = obj0_velp - grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        # to hide the "goal" sphere
        achieved_goal = grip_pos.copy()
        achieved_goal[2] += 0.1
        self.goal = achieved_goal

        obs = np.concatenate(
            [
                grip_pos,
                obj0_pos,
                grip_velp,
                obj0_velp,
                gripper_state,
                gripper_vel,
                obj0_rel_pos,
                np.zeros(3),
                np.zeros(3),
                obj0_rot,
                np.zeros(3),
                np.zeros(3),
                # obj0_velr,
                # np.zeros(3),
                # np.zeros(3),
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def step(self, action):
        # copied from RobotEnv to pass prev_obs and current obs to compute reward
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_obs = self._get_obs()
        self._set_action(action*2)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(prev_obs, obs, self.goal, info)
        return obs, reward, done, info

    def _is_within_x_bounds(self, pos):
        posxy_absdelta = np.abs(pos[0:1] - np.array([1.3]))
        return posxy_absdelta[0] < 0.125

    def compute_reward(self, prev_obs, curr_obs, goal, info):
        o1_pos = curr_obs['observation'][3:6]
        r_ret = o1_pos[1] - 1.1
        if not self._is_within_x_bounds(o1_pos):
            r_ret -= 1
        return r_ret


class FetchPushDouble(fetch_env.FetchEnv, gymutils.EzPickle):
    def __init__(self, reward_type=''):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_env.FetchEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__), "assets/robotics/fetch/pushdouble.xml"),
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.22,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.0,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        gymutils.EzPickle.__init__(self, reward_type=reward_type)
        self.init_obs = None
        assert self.has_object

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        ref_qpos = self.sim.data.get_joint_qpos("object0:joint")
        ref_qpos[:2] = self.initial_gripper_xpos[:2]
        offset = 0.03
        obj0_qpos = ref_qpos.copy()
        obj0_qpos[:2] += np.array([1, 0])*offset
        obj1_qpos = ref_qpos.copy()
        obj1_qpos[:2] += np.array([-1, 0])*offset
        self.sim.data.set_joint_qpos("object0:joint", obj0_qpos)
        self.sim.data.set_joint_qpos("object1:joint", obj1_qpos)

        self.sim.forward()
        self.init_obs = self._get_obs()
        return True

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        obj0_pos = self.sim.data.get_site_xpos("object0")
        obj0_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        obj0_velp = self.sim.data.get_site_xvelp("object0") * dt
        obj0_velr = self.sim.data.get_site_xvelr("object0") * dt

        obj1_pos = self.sim.data.get_site_xpos("object1")
        obj1_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object1"))
        obj1_velp = self.sim.data.get_site_xvelp("object1") * dt
        obj1_velr = self.sim.data.get_site_xvelr("object1") * dt

        obj0_rel_pos = obj0_pos - grip_pos
        obj0_velp -= grip_velp
        obj1_rel_pos = obj1_pos - grip_pos
        obj1_velp -= grip_velp
        objs_rel_pos = obj0_pos - obj1_pos

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        # to hide the "goal" sphere
        achieved_goal = grip_pos.copy()
        achieved_goal[2] += 0.1
        self.goal = achieved_goal

        obs = np.concatenate(
            [
                grip_pos,
                obj0_pos,
                obj1_pos,
                grip_velp,
                gripper_state,
                gripper_vel,
                obj0_rel_pos,
                obj1_rel_pos,
                objs_rel_pos,
                obj0_rot,
                obj1_rot,
                # obj0_velp,
                # obj0_velr,
                # obj1_velp,
                # obj1_velr,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def step(self, action):
        # copied from RobotEnv to pass prev_obs and current obs to compute reward
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_obs = self._get_obs()
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(prev_obs, obs, self.goal, info)
        return obs, reward, done, info

    def _has_moved_only_along_y(self, pos):
        posxy_absdelta = np.abs(pos[0:2] - np.array([1.3, 0.75]))
        return posxy_absdelta[0] < 0.125 and posxy_absdelta[1] > 0.075

    def compute_reward(self, prev_obs, curr_obs, goal, info):
        o1_pos = curr_obs['observation'][3:6]
        o2_pos = curr_obs['observation'][6:9]
        r_ret = -np.abs(o1_pos - o2_pos)[1]
        if not self._has_moved_only_along_y(o1_pos):
            r_ret -= 1
        if not self._has_moved_only_along_y(o2_pos):
            r_ret -= 1
        return r_ret


class FetchPickPlace(fetch_env.FetchEnv, gymutils.EzPickle):
    def __init__(self, reward_type=''):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_env.FetchEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__), "assets/robotics/fetch/pickplace.xml"),
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.22,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.0,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        gymutils.EzPickle.__init__(self, reward_type=reward_type)
        self.init_obs = None
        assert self.has_object

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        ref_qpos = self.sim.data.get_joint_qpos("object0:joint")
        ref_qpos[:2] = self.initial_gripper_xpos[:2]
        obj0_qpos = ref_qpos.copy()
        offset = 0.15
        tgtoffset = 0.25 # ensure this is same as the one in demo generator
        obj0_qpos[:2] += np.array([1, 0])*-offset
        self.sim.data.set_joint_qpos("object0:joint", obj0_qpos)

        self.sim.forward()
        self.init_obs = self._get_obs()
        self.tgt_obj_x = obj0_qpos[0] + tgtoffset

        return True

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        obj0_pos = self.sim.data.get_site_xpos("object0")
        obj0_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        obj0_velp = self.sim.data.get_site_xvelp("object0") * dt
        obj0_velr = self.sim.data.get_site_xvelr("object0") * dt

        obj0_rel_pos = obj0_pos - grip_pos
        obj0_rel_velp = obj0_velp - grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        # to hide the "goal" sphere
        achieved_goal = grip_pos.copy()
        achieved_goal[2] += 0.1
        self.goal = achieved_goal

        obs = np.concatenate(
            [
                grip_pos,
                obj0_pos,
                grip_velp,
                obj0_velp,
                gripper_state,
                gripper_vel,
                obj0_rel_pos,
                # np.zeros(3),
                # np.zeros(3),
                # obj0_rot,
                # np.zeros(3),
                # np.zeros(3),
                # obj0_velr,
                # np.zeros(3),
                # np.zeros(3),
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def step(self, action):
        # copied from RobotEnv to pass prev_obs and current obs to compute reward
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_obs = self._get_obs()
        self._set_action(action*2)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'obj_pos_y': obs['observation'][4],
        }
        reward = self.compute_reward(prev_obs, obs, self.goal, info)
        return obs, reward, done, info

    def compute_reward(self, prev_obs, curr_obs, goal, info):
        o1_pos = curr_obs['observation'][3:6]
        r_ret = -np.abs(o1_pos[0] - self.tgt_obj_x)
        return r_ret

class FetchPickPlaceWide(FetchPickPlace):

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        ref_qpos = self.sim.data.get_joint_qpos("object0:joint")
        ref_qpos[:2] = self.initial_gripper_xpos[:2]
        obj0_qpos = ref_qpos.copy()
        offset = 0.15
        tgtoffset = 0.20 # ensure this is same as the one in demo generator
        obj0_qpos[:2] += np.array([1, 0])*-offset
        self.sim.data.set_joint_qpos("object0:joint", obj0_qpos)

        self.sim.forward()
        self.init_obs = self._get_obs()
        self.tgt_obj_x = obj0_qpos[0] + tgtoffset

        return True
    
    def _get_obs(self):
        robot_qpos, _ = utils.robot_get_obs(self.sim)
        obs = super()._get_obs()
        obs['observation'] = np.concatenate([obs['observation'], robot_qpos])
        return obs
