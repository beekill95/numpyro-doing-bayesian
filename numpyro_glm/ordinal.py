import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd


def one_group(ranking_data: pd.DataFrame, ranks: 'list[str]'):
    nb_rows = len(ranking_data)
    nb_ranks = len(ranks)

    with numpyro.plate('Subject', nb_rows) as idx:
        # Specify mean and standard deviation of the latent continuous variable.
        mean = numpyro.sample('mean', dist.Normal(
            (nb_ranks + 1.) / 2, nb_ranks))
        std = numpyro.sample('sigma', dist.Uniform(
            nb_ranks / 1000., nb_ranks * 10.))

        # Specify latent score.
        score = dist.Normal(mean, std)

        # Then, specify the threshold for each ordinal outcome.
        thresholds = []
        probs = []
        with numpyro.plate('Rank', nb_ranks) as rank_idx:
            # Sample threshold for each ordinal outcome.
            if rank_idx == 0:
                thres = numpyro.deterministic('thres[0]', 0.5)
                thresholds.append(thres)
            elif rank_idx == nb_ranks - 1:
                thres = numpyro.deterministic(
                    f'thres[{rank_idx - 1}]', nb_ranks - 0.5)
                thresholds.append(thres)
            else:
                thres = numpyro.sample(
                    f'thres[{rank_idx}]', dist.Normal(rank_idx + 0.5, 2))
                thresholds.append(thres)

            # Calculate the probability of having an outcome.
            if rank_idx == 0:
                prob = score.cdf(thresholds[0])
                probs.append(prob)
            elif rank_idx == nb_ranks - 1:
                prob = 1 - score.cdf(thresholds[-1])
                probs.append(prob)
            else:
                prob = jnp.max(
                    0, score.cdf(thresholds[idx] - score.cdf(thresholds[idx - 1])))
                probs.append(prob)

        # Finally, specify the observed outcome.
        vote = ranking_data.iloc[idx][ranks]
        N = vote.sum()
        numpyro.sample('obs', dist.Multinomial(N, probs), obs=vote)
