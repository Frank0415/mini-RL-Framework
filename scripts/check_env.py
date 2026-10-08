import argparse
import platform

import gymnasium as gym
import numpy as np
import torch


def check_torch(device):
    tensor = torch.ones((32, 32), device=device, requires_grad=True)
    loss = (tensor @ tensor).sum()
    torch.testing.assert_close(loss, tensor.new_tensor(32768.0))
    loss.backward()
    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 64.0))
    print(f"PyTorch {device}: matrix multiplication and backward passed")


def check_cartpole():
    env = gym.make("CartPole-v1")
    try:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)
        env.action_space.seed(0)
        total_reward = 0.0
        for step in range(1, 501):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            observation, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
            assert isinstance(terminated, bool) and isinstance(truncated, bool)
            assert isinstance(info, dict)
            total_reward += reward
            if terminated or truncated:
                break
        assert terminated or truncated
        print(
            f"CartPole-v1: reset/step passed, steps={step}, "
            f"reward={total_reward}, terminated={terminated}, truncated={truncated}, "
            f"observation_shape={observation.shape}, info={info}"
        )
    finally:
        env.close()

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        print(f"CartPole-v1: headless RGB rendering passed, frame_shape={frame.shape}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Check the RL development environment.")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(
        f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
        f"CUDA runtime {torch.version.cuda}, Gymnasium {gym.__version__}, "
        f"NumPy {np.__version__}"
    )
    check_torch("cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        check_torch("cuda")
    elif args.require_cuda:
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False")
    else:
        print("CUDA unavailable; GPU check skipped")
    check_cartpole()


if __name__ == "__main__":
    main()
