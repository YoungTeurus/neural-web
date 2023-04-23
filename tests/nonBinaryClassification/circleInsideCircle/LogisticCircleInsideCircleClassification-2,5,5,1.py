from activation_funcs.LogisticFunc import Logistic
from entities.NeuralWeb import NeuralWeb
from tasks.classification.CircleInsideCircleClassification import CircleInsideCircleClassification
from trainer import Trainer2
from visualize.VisualiserHelper import VisualiserHelper

if __name__ == "__main__":
    options = CircleInsideCircleClassification.Options(0.5, 0.5, 0.05, 500)
    data = CircleInsideCircleClassification.generateInputOptions(options)

    VisualiserHelper.CircleInsideCircleClassification(data)

    nw = NeuralWeb.createNew(
        [CircleInsideCircleClassification.inputShape(),
         5,
         5,
         CircleInsideCircleClassification.outputShape()],
        [Logistic(), Logistic(), Logistic()]
    )

    Trainer2.train(nw, data, Trainer2.TrainOptions(0.15, 1500, 0))

    VisualiserHelper.CircleInsideCircleClassificationResult(nw)
