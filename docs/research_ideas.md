# DeepSensor research ideas

Are you interested in using DeepSensor for a research project?
Thankfully there are many interesting open questions with ConvNPs and their application
to environmental science.
Below are a non-exhaustive selection of research ideas that you could explore.
It would be helpful to ensure you are familiar with the literature and
resources in the [](resources.md) page before starting.

Why not [join our Slack channel](https://docs.google.com/forms/d/e/1FAIpQLScsI8EiXDdSfn1huMp1vj5JAxi9NIeYLljbEUlMceZvwVpugw/viewform)
and start a conversation around these ideas or your own? :-)

## Transfer learning from regions of dense observations to regions of sparse observations
Since the `ConvNP` is a data-hungry model, it does not perform well if only trained on a
small number of observations, which presents a challenge for modelling variables that
are poorly observed.
But what if a particular variable is well observed in one region and poorly observed in another?
Can we train a model on a region of dense observations and then transfer the model to a region
of sparse observations?
Does the performance improve?

## Sensor placement for forecasting
Previous active learning research with ConvNPs has only considered sensor placement for interpolation.
Do the sensor placements change when the model is trained for forecasting?

See, e.g., Section 4.2.1 of [Environmental sensor placement with convolutional Gaussian neural processes](https://doi.org/10.1017/eds.2023.22).

## U-Net architectural changes
The `ConvNP` currently uses a vanilla U-Net architecture.
Do any architectural changes improve performance, such as batch normalisation or dropout?

This would require digging into the [`neuralprocesses.construct_convgnp` method](https://github.com/wesselb/neuralprocesses/blob/f20572ba480c1279ad5fb66dbb89cbc73a0171c7/neuralprocesses/architectures/convgnp.py#L97)
and replacing the U-Net module with a custom one.

## Extension to continuous time observations
The `ConvNP` currently assumes that the observations are on a regular time grid.
How can we extend this to continuous time observations, where the observations are not necessarily
on a regular time grid?
Can we do this without a major rework of the code and model?
For example, can we pass a 'time of observation' auxiliary input to the model?
What are the limitations of this approach?

## Training with ablations for interpretability
Since the `ConvNP` operates on sets of observations, it is possible to ablate observations
and see how the model's predictions change.
Thus, the `ConvNP` admits unique interpretability opportunities.

However, the model would need to be trained with examples of ablated observations so that it
is not out of distribution when it sees ablated observations at test time.
For example, when generating `Task`s with a `TaskLoader`, randomly set some of the
`context_sampling` entries to `0` to remove all observations for those context sets.
Then, at test time, ablate context sets and measure the change in the model's predictions
or performance.

## Monte Carlo sensor placement using AR sampling
The `GreedyAlgorithm` for sensor placement currently uses the model's mean prediction
to infill missing observations at query sites.
However, one could also draw multiple [AR samples](user-guide/prediction.ipynb)
from the model to perform *Monte Carlo sampling* over the acquisition function.

How does this change the sensor placements and what benefits does it yield?
Do the acquisition functions become more robust (e.g. correlate better with
true performance gains)?

The [Environmental sensor placement with convolutional Gaussian neural processes](https://doi.org/10.1017/eds.2023.22)
paper will be important background reading for this.
