package org.acme;

import ai.djl.Model;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * got from: https://github.com/deepjavalibrary/djl/issues/38
 */
public class EarlyStoppingFit {
    private static final Logger logger = LoggerFactory.getLogger(EarlyStoppingFit.class);
    /**
     * done if validation loss objective (e.g. L2Loss) < threshold
     */
    private final int maxEpochs;

    private final double objectiveSuccess;

    /**
     * after minimum # epochs, consider stopping if:
     */
    private final int minEpochs;
    /**
     * too much time elapsed
     */
    private final int maxMinutes;
    /**
     * consider early stopping if not x% improvement
     */
    private final double earlyStopPctImprovement;
    /**
     * stop if insufficient improvement for x epochs in a row
     */
    private final int earlyStopPatience;

    private final List<AfterEpoch> callablesAfterEpoch = new ArrayList<>();

    public EarlyStoppingFit(int maxEpochs, double objectiveSuccess, int minEpochs, int maxMinutes, double earlyStopPctImprovement, int earlyStopPatience) {
        this.maxEpochs = maxEpochs;
        this.objectiveSuccess = objectiveSuccess;
        this.minEpochs = minEpochs;
        this.maxMinutes = maxMinutes;
        this.earlyStopPctImprovement = earlyStopPctImprovement;
        this.earlyStopPatience = earlyStopPatience;
    }

    public void addCallback(AfterEpoch afterEpoch) {
        callablesAfterEpoch.add(afterEpoch);
    }

    public void fit(Trainer trainer, RandomAccessDataset trainingSet, RandomAccessDataset validateSet) throws TranslateException, IOException {
        final long start = System.currentTimeMillis();
        double prevLoss = Double.NaN;
        int improvementFailures = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            for (Batch batch : trainer.iterateDataset(trainingSet)) {
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
                batch.close();
            }

            // After each epoch, test against the validation dataset if we have one
            EasyTrain.evaluateDataset(trainer, validateSet);

            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));

            final int currentEpoch = epoch + 1;
            final double vloss = trainer.getTrainingResult().getValidateLoss();// else use train loss if no validation set
            callablesAfterEpoch.forEach(callable -> callable.call(trainer.getModel(), currentEpoch, vloss));

            // stopping criteria
            if (vloss < objectiveSuccess) {
                logger.info("END: validation loss {} < objectiveSuccess {}", vloss, objectiveSuccess);
                return;
            }
            logger.info("On epoch {}, validation loss {}", currentEpoch, vloss);
            if (epoch + 1 >= minEpochs) {
                double elapsedMinutes = (System.currentTimeMillis() - start) / 60_000.0;
                if (elapsedMinutes >= maxMinutes) {
                    logger.info("END: %.1f minutes elapsed >= %s maxMinutes".formatted(elapsedMinutes, maxMinutes));
                    return;
                }
                // consider early stopping?
                if (Double.isFinite(prevLoss)) {
                    double goalImprovement = prevLoss * (100 - earlyStopPctImprovement) / 100.0;
                    boolean improved = vloss <= goalImprovement;// false if any NANs
                    if (improved) {
                        improvementFailures = 0;
                    } else {
                        improvementFailures++;
                        if (improvementFailures >= earlyStopPatience) {
                            logger.info("END: failed to achieve %s%% improvement %s times in a row".formatted(
                                    earlyStopPctImprovement, earlyStopPatience));
                            return;
                        }
                    }
                }
            }
            if (Double.isFinite(vloss)) {
                prevLoss = vloss;
            }
        }
    }

    public interface AfterEpoch {
        void call(Model model, int epoch, double validationLoss);
    }
}
