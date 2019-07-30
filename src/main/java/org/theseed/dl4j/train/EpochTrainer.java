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
import org.theseed.dl4j.train.TrainingProcessor.RunStats;

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
    public EpochTrainer(TrainingProcessor processor, Logger log) {
        super(processor, log);
    }

    @Override
    public RunStats trainModel(MultiLayerNetwork model, Iterator<DataSet> reader) {
        RunStats retVal = new RunStats(model);
        // Get all of the batches into a list, up to the maximum.
        log.info("Reading training data into memory.");
        List<DataSet> batches = new ArrayList<DataSet>();
        int batchesRead;
        for (batchesRead = 0; batchesRead < this.processor.getMaxBatches() && reader.hasNext(); batchesRead++)
            batches.add(reader.next());
        // Initialize the old score for bounce detection.
        double oldScore = Double.MAX_VALUE;
        // Initialize the stats and counters for keeping the best generation.
        double bestScore = Double.MAX_VALUE;
        int bestIter = 0;
        int numSaves = 0;
        // Do one epoch per iteration.
        while (retVal.getEventCount() < processor.getIterations() && ! retVal.isErrorStop()) {
            retVal.event();
            log.info("Processing epoch {}.", retVal.getEventCount());
            long start = System.currentTimeMillis();
            for (DataSet batch : batches) {
                model.fit(batch);
            }
            double seconds = (double) (System.currentTimeMillis() - start) / 1000;
            double newScore = model.score();
            String saveFlag = "";
            if (newScore > oldScore) {
                retVal.bounce();
            } else if (newScore < bestScore) {
                retVal.setBestModel(model.clone());
                bestScore = newScore;
                bestIter = retVal.getEventCount();
                saveFlag = "  Model saved.";
                numSaves++;
            }
            oldScore = newScore;
            log.info("Score after epoch {} is {}.  {} seconds to process {} batches.{}", retVal.getEventCount(),
                    newScore, seconds, batchesRead, saveFlag);
            // Force a stop if we have overflow or underflow.
            if (! Double.isFinite(newScore)) {
                log.error("Overflow/Underflow in gradient processing.  Model abandoned.");
                retVal.error();
            }
        }
        log.info("Best model was epoch {} with score {}.  {} models saved.", bestIter, bestScore, numSaves);
        return retVal;
    }

    @Override
    public String eventsName() {
        return "epochs";
    }

}