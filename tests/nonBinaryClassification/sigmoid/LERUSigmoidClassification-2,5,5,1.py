from activation_funcs.LERU import LERU
from entities.NeuralWeb import NeuralWeb
from tasks.classification.SigmoidBinaryClassification import SigmoidBinaryClassification
from trainer import Trainer2
from visualize.VisualiserHelper import VisualiserHelper

if __name__ == "__main__":
    options = SigmoidBinaryClassification.Options(0.5, 0.3, 8, 250)
    data = SigmoidBinaryClassification.generateInputOptions(options)

    VisualiserHelper.SigmoidBinaryClassification(data)

    nw = NeuralWeb.createNew(
        [SigmoidBinaryClassification.inputShape(),
         5,
         5,
         SigmoidBinaryClassification.outputShape()],
        [LERU(), LERU(), LERU()]
    )

    Trainer2.train(nw, data, Trainer2.TrainOptions(0.15, 1000, 0))

    VisualiserHelper.SigmoidBinaryClassificationResult(nw)
