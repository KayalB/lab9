# Common hyperparameters for all milestones
CONFIG = {
    "env": {
        "N": 6,
        "M": 6,
        "wall_prob": 0.1
    },
    "q_learning": {
        "alpha": 0.1,
        "gamma": 0.9,
        "episodes": 1000,
        "max_steps": 1000,
        "epsilon_start": 0.9,
        "epsilon_end": 0.1
    },
    "dqn": {
        "alpha": 0.001,
        "gamma": 0.9,
        "buffer_size": 1000,
        "batch_size": 32,
        "epsilon_start": 0.9,
        "epsilon_end": 0.1
    }
}
