from activation_funcs.LogisticFunc import Logistic
from entities.NeuralWeb import NeuralWeb
from tasks.classification.LinearBinaryClassification import LinearBinaryClassification
from trainer import Trainer2
from trainer.Trainer2 import TrainOptions
from visualize.VisualiserHelper import VisualiserHelper


def printAandB(nw: NeuralWeb) -> None:
    neuron = nw.layers[0][0]
    w1, w2, w3 = neuron.weights
    a = -w1 / w2
    b = -w3 / w2

    print("{:.2f}x{}".format(a, "{:.2f}".format(b) if b < 0 else "+{:.2f}".format(b)))

if __name__ == "__main__":
    options = LinearBinaryClassification.Options.default()
    data = LinearBinaryClassification.generateInputOptions(options)

    nw = NeuralWeb.createNew(
        [LinearBinaryClassification.inputShape(),
         LinearBinaryClassification.outputShape()],
        [Logistic()]
    )

    printAandB(nw)

    Trainer2.train(nw, data, detailedLogs=False, trainOptions=TrainOptions(0.1, 1000, 0.0001), doOnEachEpoch=printAandB)

    VisualiserHelper.LinearBinaryClassificationResult(options, data, nw)