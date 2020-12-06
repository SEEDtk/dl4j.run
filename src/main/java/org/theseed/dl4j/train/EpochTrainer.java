/**
 *
 */
package org.theseed.dl4j.train;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;

/**
 * The epoch trainer reads the entire input file into memory, then runs all the batches through in an epoch
 * and goes back to do this again multiple times.  It provides better results for smaller datasets.
 * The best-scoring model is saved.
 *
 * @author Bruce Parrello
 *
 */
public class EpochTrainer extends Trainer {

    /**
     * Construct an epoch trainer.
     *
     * @param processor		TrainingProcessor that created this model
     * @param log			log to use for messages
     */
    public EpochTrainer(LearningProcessor processor, Logger log) {
        super(processor, log);
    }

    @Override
    public void trainModel(MultiLayerNetwork model, Iterator<DataSet> reader, DataSet testingSet, RunStats runStats, ITrainReporter monitor) {
        // Get all of the batches into a list, up to the maximum.
        log.info("Reading training data into memory.");
        List<DataSet> batches = new ArrayList<DataSet>();
        int batchesRead;
        for (batchesRead = 0; batchesRead < this.processor.getMaxBatches() && reader.hasNext(); batchesRead++)
            batches.add(reader.next());
        String process = batchesRead + " batches";
        // Initialize the old score for bounce detection.
        double oldScore = Double.MAX_VALUE;
        // Do one epoch per iteration.
        while (runStats.getEventCount() < processor.getIterations() && ! runStats.isErrorStop() &&
                runStats.getUselessIterations() < processor.getEarlyStop()) {
            runStats.event();
            long start = System.currentTimeMillis();
            for (DataSet batch : batches) {
                model.fit(batch);
            }
            double seconds = (double) (System.currentTimeMillis() - start) / 1000;
            double newScore = model.score();
            boolean saved = false;
             if (newScore > oldScore) {
                runStats.bounce();
                log.info("Score after {} epochs is {}.  {} seconds to process {}.", runStats.getEventCount(),
                        newScore, seconds, process);
                runStats.uselessIteration();
            } else try {
                saved = runStats.checkModel(model, testingSet, this.processor, seconds, this.eventsName(), process);
            } catch (IllegalStateException e) {
                // Here we had underflow in the evaluation.  Fake a score bounce.
                log.warn("IllegalStateException: {}", e.getMessage());
                runStats.error();
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
        log.info("Best model was epoch {} with score {}.  {} models saved.", runStats.getBestEvent(),
                runStats.getBestScore(), runStats.getSaveCount());
    }


    @Override
    public String eventsName() {
        return "epochs";
    }

}
