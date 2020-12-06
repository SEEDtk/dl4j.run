/**
 *
 */
package org.theseed.dl4j.train;

import java.util.Iterator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;

/**
 * This trainer processes the dataset in batch mode, one batch at a time with many iterations.
 * It is the preferred method for very large datasets.
 *
 * @author Bruce Parrello
 *
 */
public class BatchTrainer extends Trainer {

    /**
     * Construct a batch trainer.
     *
     * @param processor		TrainingProcessor that created this model
     * @param log			log to use for messages
     */
    public BatchTrainer(LearningProcessor processor, Logger log) {
        super(processor, log);
    }

    /**
     * Train the model one batch at a time.
     *
     * @param model			the model to train
     * @param reader		reader for traversing the training data
     * @param testingSet	testing set for evaluation
     * @param runStats		a RunStats object describing our progress and success
     * @param monitor		ITrainReporter for progress monitoring
     */
    @Override
    public void trainModel(MultiLayerNetwork model, Iterator<DataSet> reader, DataSet testingSet, RunStats runStats, ITrainReporter monitor) {
        double oldScore = Double.MAX_VALUE;
        String process = processor.getIterations() + " iterations";
        while (reader.hasNext() && runStats.getEventCount() < processor.getMaxBatches() && ! runStats.isErrorStop()) {
            // Record this batch.
            runStats.event();
            // Read it in and train with it.
            long startTime = System.currentTimeMillis();
            DataSet trainingData = reader.next();
            for(int i=0; i < processor.getIterations(); i++ ) {
                model.fit(trainingData);
            }
            long duration = (System.currentTimeMillis() - startTime) / 1000;
            // Check for a score bounce.
            boolean saved = false;
            double newScore = model.score();
            if (oldScore < newScore) {
                runStats.bounce();
                log.info("Score at end of batch {} is {}.", runStats.getEventCount(),
                        newScore);
            } else try {
                saved = runStats.checkModel(model, testingSet, this.processor, duration, this.eventsName(), process);
            } catch (IllegalStateException e) {
                // Here we had underflow in the evaluation.
                newScore = Double.NaN;
                log.warn("IllegalStateException: {}", e.getMessage());
            }
            oldScore = newScore;
            // Force a stop if we have overflow or underflow.
            if (! Double.isFinite(newScore)) {
                log.error("Overflow/Underflow in gradient processing.  Model abandoned.");
                runStats.error();
            } else {
                monitor.displayEpoch(runStats.getEventCount(), newScore, saved);
            }
        }
    }

    @Override
    public String eventsName() {
        return "batches";
    }

}
