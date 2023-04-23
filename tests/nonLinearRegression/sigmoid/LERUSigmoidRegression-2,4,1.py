from activation_funcs.LERU import LERU
from entities.NeuralWeb import NeuralWeb
from tasks.regression.SigmoidRegression import SigmoidRegression
from trainer import Trainer2
from visualize.VisualiserHelper import VisualiserHelper

if __name__ == "__main__":
    options = SigmoidRegression.Options(0.5, 0.5, 10, 50)
    data = SigmoidRegression.generateInputOptions(options)

    VisualiserHelper.XToY(data)

    nw = NeuralWeb.createNew(
        [SigmoidRegression.inputShape(),
         4,
         SigmoidRegression.outputShape()],
        [LERU(), LERU()]
    )

    Trainer2.train(nw, data, Trainer2.TrainOptions(0.1, 10000, 0))

    VisualiserHelper.SigmoidRegressionResult(nw, data)