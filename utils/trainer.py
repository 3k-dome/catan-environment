# type: ignore

from tf_agents.agents import TFAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from agent.parameters import AgentParams


def train_offline(agent: TFAgent, buffer: TFUniformReplayBuffer, parameters: AgentParams, batches: int) -> list[float]:
    loss_info: list[float] = []
    dataset = buffer.as_dataset(
        num_parallel_calls=4,
        sample_batch_size=parameters.batchsize,
        num_steps=parameters.batchsize,
    )

    batch_iterator = iter(dataset)
    for _ in range(batches):
        batch, _ = next(batch_iterator)
        loss = agent.train(batch)
        loss_info.append(float(loss.loss))

    return loss_info
