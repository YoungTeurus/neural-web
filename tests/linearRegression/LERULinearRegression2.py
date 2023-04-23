from activation_funcs.LERU import LERU
from entities.NeuralWeb import NeuralWeb
from tasks.regression.LinearRegression import LinearRegression
from trainer import Trainer2
from visualize.VisualiserHelper import VisualiserHelper


if __name__ == "__main__":
    options = LinearRegression.Options.default()
    data = LinearRegression.generateInputOptions(options)

    VisualiserHelper.LinearRegression(options, data)

    nw = NeuralWeb.createNew(
        [LinearRegression.inputShape(),
         LinearRegression.outputShape()],
        [LERU()]
    )

    Trainer2.train(nw, data, Trainer2.TrainOptions(0.1, 1000, 0.00001))

    VisualiserHelper.LinearRegressionResult(nw, options, data)


