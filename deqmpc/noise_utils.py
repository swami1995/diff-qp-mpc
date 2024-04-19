####################
# NOISE UTILS
####################
import torch


def corrupt_observation(states, noise_type, noise_std, noise_mean):
    if noise_type == 0:
        # no noise
        return states
    elif noise_type == 1:
        # gaussian noise
        noise = torch.randn_like(states) * noise_std + noise_mean
        return states + noise
    elif noise_type == 2:
        # uniform noise
        noise = 2*(torch.rand_like(states)-0.5) * noise_std + noise_mean
        return states + noise
    elif noise_type == 3:
        # any state element can be dropped to 0
        mask = torch.rand_like(states) > noise_mean
        masked_states = 1.0*states
        masked_states[mask == False] = 0.0
        return masked_states
    elif noise_type == 4:
        # any state vector can be dropped to 0
        mask = torch.rand(states.shape[:2]) > noise_mean
        mask = mask.unsqueeze(-1).repeat(1, 1, states.shape[2])
        masked_states = 1.0*states
        # only mask dimensions 0 and 1 of masked_states
        masked_states[mask == False] = 0.0
        return masked_states
    elif noise_type == 5:
        # any state element can be dropped to previous value
        mask = torch.rand_like(states) > noise_mean
        masked_states = 1.0*states
        masked_states[mask == False] = torch.roll(masked_states, 1, 1)[mask == False]
        return masked_states
    elif noise_type == 6:
        # any state vector can be dropped to previous value 
        mask = torch.rand(states.shape[:2]) > noise_mean
        mask = mask.unsqueeze(-1).repeat(1, 1, states.shape[2])
        masked_states = 1.0*states
        # only mask dimensions 0 and 1 of masked_states
        masked_states[mask == False] = torch.roll(masked_states, 1, 1)[mask == False]
        return masked_states
    else:
        raise NotImplementedError


# run main
if __name__ == "__main__":
    bsz = 3
    T = 5
    nx = 2
    states = torch.rand((bsz, T, nx), dtype=torch.float64)
    noise_type = 6
    noise_std = 0.0
    noise_mean = 0.2
    corrupted_states = corrupt_observation(
        states, noise_type, noise_std, noise_mean)
    print("Corrupted states")
    print(corrupted_states)
    print("States")
    print(states)
