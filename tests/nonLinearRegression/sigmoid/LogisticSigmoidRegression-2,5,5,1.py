from activation_funcs.LogisticFunc import Logistic
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
         5,
         5,
         SigmoidRegression.outputShape()],
        [Logistic(), Logistic(), Logistic()]
    )

    Trainer2.train(nw, data, Trainer2.TrainOptions(0.1, 5000, 0))

    VisualiserHelper.SigmoidRegressionResult(nw, data)
