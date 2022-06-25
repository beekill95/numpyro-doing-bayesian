import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd


def one_group(ranking_data: pd.DataFrame, ranks: 'list[str]'):
    nb_rows = len(ranking_data)
    nb_ranks = len(ranks)

    with numpyro.plate('Subject', nb_rows) as idx:
        # Specify mean and standard deviation of the latent continuous variable.
        mean = numpyro.sample(
            'mean', dist.Normal((nb_ranks + 1.) / 2, nb_ranks))
        std = numpyro.sample(
            'sigma', dist.Uniform(nb_ranks / 1000., nb_ranks * 10.))

        # Specify latent score.
        score = dist.Normal(mean, std)

        # Then, specify the threshold for each ordinal outcome.
        thresholds = [
            numpyro.sample('-inf', dist.Delta(-jnp.inf), obs=-jnp.inf),
            numpyro.sample('thres[0]', dist.Delta(0.5), obs=0.5),
            *[numpyro.sample(f'thres[{i}]', dist.Normal(i + 0.5, 2))
              for i in range(1, nb_ranks - 2)],
            numpyro.sample(f'thres[{nb_ranks}]',
                           dist.Delta(nb_ranks - 0.5), obs=nb_ranks - 0.5),
            numpyro.sample('inf', dist.Delta(jnp.inf), obs=jnp.inf),
        ]
        thresholds = jnp.array(thresholds)
        probs = score.cdf(thresholds)
        # print(probs.shape)
        probs = jnp.maximum(probs[1:] - probs[:-1], 0)
        # print('after', probs.shape)
        # for i in range(nb_ranks):
        #     if i == 0:
        #         probs.append(score.cdf(thresholds[i]))
        #     elif i == nb_ranks - 1:
        #         probs.append(1 - score.cdf(thresholds[i]))
        #     else:
        #         probs.append(jnp.max(
        #             score.cdf(thresholds[i]) - score.cdf(thresholds[i - 1]),
        #             initial=0.0,
        #             keepdims=True,
        #         ))

        # with numpyro.plate('Rank', nb_ranks) as rank_idx:
        #     # Sample threshold for each ordinal outcome.
        #     if rank_idx == 0:
        #         thres = numpyro.deterministic('thres[0]', 0.5)
        #         thresholds.append(thres)
        #     elif rank_idx == nb_ranks - 1:
        #         thres = numpyro.deterministic(
        #             f'thres[{rank_idx - 1}]', nb_ranks - 0.5)
        #         thresholds.append(thres)
        #     else:
        #         thres = numpyro.sample(
        #             f'thres[{rank_idx}]', dist.Normal(rank_idx + 0.5, 2))
        #         thresholds.append(thres)

        #     # Calculate the probability of having an outcome.
        #     if rank_idx == 0:
        #         prob = score.cdf(thresholds[0])
        #         probs.append(prob)
        #     elif rank_idx == nb_ranks - 1:
        #         prob = 1 - score.cdf(thresholds[-1])
        #         probs.append(prob)
        #     else:
        #         prob = jnp.max(
        #             0, score.cdf(thresholds[idx] - score.cdf(thresholds[idx - 1])))
        #         probs.append(prob)

        # Finally, specify the observed outcome.
        # probs = jnp.array(probs)
        probs = probs / probs.sum()
        vote = ranking_data[idx]
        N = vote.sum()
        numpyro.sample(
            'obs', dist.MultinomialProbs(total_count=N, probs=probs.T), obs=vote)


def one_group_1(ranking_data: pd.DataFrame, ranks: 'list[str]'):
    nb_rows = len(ranking_data)
    nb_ranks = len(ranks)

    with numpyro.plate('Subject', nb_rows) as idx:
        # Specify mean and standard deviation of the latent continuous variable.
        mean = numpyro.sample(
            'mean', dist.Normal((nb_ranks + 1.) / 2, nb_ranks))
        std = numpyro.sample(
            'sigma', dist.Uniform(nb_ranks / 1000., nb_ranks * 10.))

        # Specify latent score.
        score = dist.Normal(mean, std)

        # Then, specify the threshold for each ordinal outcome.
        cutpoints = [
            numpyro.sample('thres_0', dist.Normal(
                0.5, nb_ranks), obs=jnp.array([0.5])),
            numpyro.sample('thres_1', dist.Normal(1.5, nb_ranks)),
            numpyro.sample('thres_2', dist.Normal(2.5, nb_ranks)),
            numpyro.sample('thres_3', dist.Normal(3.5, nb_ranks)),
            numpyro.sample('thres_4', dist.Normal(
                4.5, nb_ranks), obs=jnp.array([4.5])),
        ]
        cutpoints = jnp.array(cutpoints)
        cdf = score.cdf(cutpoints)

        probs = jnp.zeros((6, 1), dtype=jnp.float32)
        probs = probs.at[0].set(cdf[0])
        probs = probs.at[jnp.array([1, 2, 3, 4])].set(cdf[1:] - cdf[:-1])
        probs = probs.at[5].set(1 - cdf[-1])
        probs = jnp.maximum(probs, 0.0)

        # Finally, specify the observed outcome.
        # probs = jnp.array(probs)
        probs = probs / probs.sum()
        vote = ranking_data[idx]
        N = vote.sum()
        numpyro.sample(
            'obs', dist.MultinomialProbs(total_count=N, probs=probs.T), obs=vote)
