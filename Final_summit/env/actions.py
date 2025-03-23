# env/actions.py
import numpy as np
from gymnasium import spaces

def setup_spaces(env):
    # Không gian quan sát
    upper = np.full(shape=(env.max_w, env.max_h), fill_value=env.max_product_type + 2, dtype=int)
    lower = np.full(shape=(env.max_w, env.max_h), fill_value=-2, dtype=int)
    observation_space = spaces.Dict(
        {
            "stocks": spaces.Tuple(
                [spaces.MultiDiscrete(upper, start=lower)] * env.num_stocks, seed=env.seed
            ),
            "products": spaces.Sequence(
                spaces.Dict(
                    {
                        "size": spaces.MultiDiscrete(
                            np.array([env.max_w, env.max_h]), start=np.array([1, 1])
                        ),
                        "quantity": spaces.Discrete(env.max_product_per_type + 1, start=0),
                    }
                ),
                seed=env.seed,
            ),
        }
    )

    # Không gian hành động
    action_space = spaces.Dict(
        {
            "stock_idx": spaces.Discrete(env.num_stocks),
            "size": spaces.Box(
                low=np.array([1, 1]),
                high=np.array([env.max_w, env.max_h]),
                shape=(2,),
                dtype=int,
            ),
            "position": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([env.max_w - 1, env.max_h - 1]),
                shape=(2,),
                dtype=int,
            ),
        }
    )

    return observation_space, action_space