from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    rewards: list[list[float]]
    steps: list[int]
    lengths: list[float]
    epsilon: float

    no_episodes: int = 0
    total_steps: int = 0
    avg_steps: float = 0.0
    med_steps: float = 0.0

    total_reward: float = 0
    avg_reward: float = 0.0
    med_reward: float = 0.0
    completed: float = 0.0

    closest_rewards: list[float] = field(default_factory=list)
    avg_closest_reward: float = 0
    med_closest_reward: float = 0
    min_closest_reward: float = 0
    max_closest_reward: float = 0
    completed_reversed: float = 0

    total_length: float = 0
    avg_length: float = 0
    med_length: float = 0

    def __post_init__(self):
        # general episode metrics
        self.no_episodes = self.steps.__len__()
        self.total_steps = sum(self.steps)
        self.min_steps = min(self.steps)
        self.max_steps = max(self.steps)
        self.avg_steps = self.total_steps / self.no_episodes
        self.med_steps = sorted(self.steps)[self.no_episodes // 2]

        # general goal metrics
        summed_rewards = [sum(rewards) for rewards in self.rewards]
        self.total_reward = sum(summed_rewards)
        self.min_reward = min(summed_rewards)
        self.max_reward = max(summed_rewards)
        self.avg_reward = self.total_reward / self.no_episodes
        self.med_reward = sorted(summed_rewards)[self.no_episodes // 2]
        self.completed = sum([1 if reward >= 1 else 0 for reward in summed_rewards]) / self.no_episodes

        # general goal on end metrics
        self.closest_rewards = [max(rewards) for rewards in self.rewards]
        self.min_closest_reward = min(self.closest_rewards)
        self.max_closest_reward = max(self.closest_rewards)
        self.avg_closest_reward = sum(self.closest_rewards) / self.no_episodes
        self.med_closest_reward = sorted(self.closest_rewards)[self.no_episodes // 2]
        self.completed_reversed = sum([1 if reward == 0 else 0 for reward in self.closest_rewards]) / self.no_episodes

        # timing metrics
        self.total_length = sum(self.lengths)
        self.avg_length = self.total_length / self.no_episodes
        self.med_length = sorted(self.lengths)[self.no_episodes // 2]

    @staticmethod
    def header() -> str:
        return ",".join(
            [
                "Epsilon",
                "Episodes",
                "Total Steps",
                "Min. Steps/Episode",
                "Max. Steps/Episode",
                "Avg. Steps/Episode",
                "Med. Steps/Episode",
                "Total Reward",
                "Min. Episode Reward",
                "Max. Episode Reward",
                "Avg. Episode Reward",
                "Med. Episode Reward",
                "Completed Games",
                "Min. Highest inEpisode Reward",
                "Max. Highest inEpisode Reward",
                "Avg. Highest inEpisode Reward",
                "Med. Highest inEpisode Reward",
                "Completed Games Reversed",
                "Total Time Spent",
                "Avg. Time Spent per Episode",
                "Total Time Spent per Episode",
            ]
        )

    def __repr__(self) -> str:
        return ",".join(
            [
                str(x)
                for x in [
                    self.epsilon,
                    self.no_episodes,
                    self.total_steps,
                    self.min_steps,
                    self.max_steps,
                    self.avg_steps,
                    self.med_steps,
                    self.total_reward,
                    self.min_reward,
                    self.max_reward,
                    self.avg_reward,
                    self.med_reward,
                    self.completed,
                    self.min_closest_reward,
                    self.max_closest_reward,
                    self.avg_closest_reward,
                    self.med_closest_reward,
                    self.completed_reversed,
                    self.total_length,
                    self.avg_length,
                    self.med_length,
                ]
            ]
        )
