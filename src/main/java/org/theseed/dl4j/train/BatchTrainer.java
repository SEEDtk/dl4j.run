/**
 *
 */
package org.theseed.dl4j.train;

import java.util.Iterator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.theseed.dl4j.train.TrainingProcessor.RunStats;

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
    public BatchTrainer(TrainingProcessor processor, Logger log) {
        super(processor, log);
    }

    //@Override
    /**
     * Train the model one batch at a time.
     *
     * @param model			the model to train
     * @param reader		reader for traversing the training data
     * @param testingSet	testing set for evaluation
     *
     * @return a RunStats object describing our progress and success
     */
    public RunStats trainModel(MultiLayerNetwork model, Iterator<DataSet> reader, DataSet testingSet) {
        RunStats retVal = new RunStats(model);
        double oldScore = Double.MAX_VALUE;
        while (reader.hasNext() && retVal.getEventCount() < processor.getMaxBatches() && ! retVal.isErrorStop()) {
            // Record this batch.
            retVal.event();
            // Read it in and train with it.
            long startTime = System.currentTimeMillis();
            log.info("Reading data batch {}.", retVal.getEventCount());
            DataSet trainingData = reader.next();
            for(int i=0; i < processor.getIterations(); i++ ) {
                model.fit(trainingData);
            }
            // Check for a score bounce.
            double newScore = model.score();
            if (oldScore < newScore) retVal.bounce();
            oldScore = newScore;
            long duration = (System.currentTimeMillis() - startTime) / 1000;
            log.info("Score at end of batch {} is {}. {} seconds for {} iterations.", retVal.getEventCount(),
                    newScore, duration, processor.getIterations());
            // Force a stop if we have overflow or underflow.
            if (! Double.isFinite(newScore)) {
                log.error("Overflow/Underflow in gradient processing.  Model abandoned.");
                retVal.error();
            }
        }
        return retVal;
    }

    @Override
    public String eventsName() {
        return "batches";
    }

}
