from activation_funcs.LogisticFunc import Logistic
from entities.NeuralWeb import NeuralWeb
from tasks.classification.LinearBinaryClassificationWithTwoOutputNeurons import LinearBinaryClassificationWithTwoOutputNeurons
from trainer import Trainer2
from trainer.Trainer2 import TrainOptions
from visualize.VisualiserHelper import VisualiserHelper

if __name__ == "__main__":
    options = LinearBinaryClassificationWithTwoOutputNeurons.Options(1, 0, 100)
    data = LinearBinaryClassificationWithTwoOutputNeurons.generateInputOptions(options)

    VisualiserHelper.LinearBinaryClassificationWithTwoOutputNeurons(options, data)

    nw = NeuralWeb.createNew(
        [LinearBinaryClassificationWithTwoOutputNeurons.inputShape(),
         3,
         LinearBinaryClassificationWithTwoOutputNeurons.outputShape()],
        [Logistic(), Logistic()]
    )

    Trainer2.train(nw, data, detailedLogs=False, trainOptions=TrainOptions(0.5, 1000, 0))

    VisualiserHelper.LinearBinaryClassificationWithTwoOutputNeuronsResult(options, nw)

